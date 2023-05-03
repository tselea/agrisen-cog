import argparse
import math
import os
import time

import dask.array as da
import datashader as ds
import geopandas as gpd
import numpy as np
import pandas as pd
import spatialpandas as sp
import stackstac
import xarray as xr
import yaml
from pystac_client import Client
from rasterio.enums import MergeAlg
from rasterio.features import rasterize as rio_rasterize
from shapely.geometry import box

from lpis_processing import config_s3 as config
from lpis_processing.cluster_connect import cluster_dispatcher
from lpis_processing.utils import read_parquet, SingleDispatcher

SELECTED_TILES_FILE = 'ds_processing/aoi_files/sample_tiles.yaml'
STAC_CATALOGUE_URL = 'https://earth-search.aws.element84.com/v0'
max_cloud = 30
datetime = "2020-07-01T00:00:00Z/2021-08-01T00:00:00Z"

key = config.key
secret = config.secret
endpoint_url = config.endpoint_url
bucket_name = config.bucket_name

import logging

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger("tile_processing")

write_dispatcher = SingleDispatcher()

raster_dispatcher = SingleDispatcher()


@write_dispatcher.register('tiff')
def write_geotiff(data_xr, output_file):
    data_xr.rio.to_raster(output_file)


@write_dispatcher.register('zarr')
def write_zarr(data_xr, output_file):
    data_xr.data = da.from_array(data_xr.data, chunks=(1024, 1024))
    data_ds = data_xr.to_dataset(name='lpis')
    data_ds.to_zarr(output_file,
                    storage_options={
                        "key": key,
                        "secret": secret,
                        "client_kwargs": {"endpoint_url": endpoint_url}
                    })


@raster_dispatcher.register('rasterio')
def raster_rasterio(s2_stack, crops_gpd, column):
    geom_value = ((geom, value) for geom, value in zip(crops_gpd.geometry, crops_gpd[column]))
    rasterized = rio_rasterize(geom_value,
                               out_shape=s2_stack.shape[2:],
                               transform=s2_stack.transform,
                               all_touched=True,
                               fill=0,  # background value
                               merge_alg=MergeAlg.replace,
                               dtype=np.int32)

    return xr.DataArray(rasterized, dims=['y', 'x'])


@raster_dispatcher.register('datashader')
def raster_datashader(s2_stack, crops_gpd, column):
    minx, miny, maxx, maxy = s2_stack.spec.bounds
    plot_width = math.ceil((maxx - minx) / 10)
    plot_height = math.ceil((maxy - miny) / 10)

    canvas = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=[minx, maxx],
                       y_range=[miny, maxy])

    crop_spd = sp.GeoDataFrame(crops_gpd)
    data_xr = canvas.polygons(crop_spd, 'geometry', agg=ds.last(column))

    if float(s2_stack.x[0]) > float(s2_stack.x[-1]):
        data_xr.data = np.flip(data_xr.data, axis=1)
    if float(s2_stack.y[0]) > float(s2_stack.y[-1]):
        data_xr.data = np.flip(data_xr.data, axis=0)

    return data_xr


def rasterize(aoi_tile, year, input_file, is_s3, column, output_dirs, output_formats, raster_method):
    # get the bounding box for each tile
    root_catalog = Client.open(STAC_CATALOGUE_URL)
    aoi_dgpd = read_parquet(input_file, is_s3, key, secret, endpoint_url)
    if (column not in aoi_dgpd.columns) and (column == 'index'):
        aoi_dgpd['index'] = aoi_dgpd.index

    aoi_dgpd_new = None

    start = time.time()

    for tile in aoi_tile:
        utm_zone = int(tile[:2])
        latitude_band = tile[2]
        grid_square = tile[3:]
        query_params = {'query':
                            {"sentinel:utm_zone": {"eq": utm_zone}, "sentinel:latitude_band": {"eq": latitude_band},
                             'sentinel:grid_square': {"eq": grid_square},
                             "eo:cloud_cover": {"lte": max_cloud}
                             },
                        'collections': ["sentinel-s2-l2a-cogs"],
                        'datetime': datetime,
                        'max_items': 2000,
                        }
        items = root_catalog.search(**query_params).get_all_items()
        s2_stack = stackstac.stack(items, resolution=10, assets=['B02'], chunksize=(-1, 1, 1024, 1024))

        tile_gpd = gpd.GeoDataFrame(pd.DataFrame({'tile_name': ['tile'], 'geometry': [box(*s2_stack.spec.bounds)]}),
                                    geometry='geometry', crs=s2_stack.crs)

        if (aoi_dgpd_new is None) or (str(aoi_dgpd_new.crs) != s2_stack.crs):
            aoi_dgpd_new = aoi_dgpd.to_crs(s2_stack.crs)
        crops_dgpd = aoi_dgpd_new.sjoin(tile_gpd, how="inner", predicate='intersects')
        crops_gpd = crops_dgpd.compute()

        data_xr = raster_dispatcher(raster_method, s2_stack, crops_gpd, column)

        data_xr = data_xr.fillna(0)
        data_xr = data_xr.astype(crops_gpd.dtypes[column])

        data_xr = data_xr.assign_coords(x=s2_stack.x.data, y=s2_stack.y.data)

        data_xr.rio.write_crs(s2_stack.crs, inplace=True)

        for output_dir, output_format in zip(output_dirs, output_formats):
            output_file = f'{output_dir}/LPIS_{year}_{tile}.{output_format}'
            write_dispatcher(output_format, data_xr, output_file)

        log.info(f'Rasterized tile {tile}.')

    end = time.time()
    print(end - start)


def main():
    # Parse user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--connect_cluster', type=str, default=None, required=False,
                        choices=['uvt', 'coiled'],
                        help='Connect to dask cluster. One of [\'uvt\']')
    parser.add_argument('--aoi', default=None, required=True,
                        help='The AOI to load selected tiles')
    parser.add_argument('--year', default=None, required=True, type=int,
                        help='The year for the rasterization.')
    parser.add_argument('--file_to_raster', default=None, required=True,
                        help='The gpd file to perform rasterization on.')
    parser.add_argument('--s3', action='store_true', default=False, required=False,
                        help='Save data on S3. Available only with output Parquet format.')
    parser.add_argument('--column', default=None, required=False,
                        help='Column values to use for rasterization.')
    parser.add_argument('--output_dirs', type=str, default=None, required=None, nargs='+',
                        help='The output dirs to write the rasterized data. For each output format a dir must be provided')
    parser.add_argument('--output_formats', type=str, default=None, required=None, nargs='+',
                        help='Output file type to save the data. Options are [zarr, tiff].')
    parser.add_argument('--input_dir', type=str, default=None, required=None,
                        help='The input dir to read LPIS and fix')

    parser.add_argument('--action', type=str, default='rasterio', required=True,
                        choices=['datashader', 'rasterio'],
                        help='The action to be performed. One of [\'datashader\', \'rasterio\']')

    args = parser.parse_args()

    if args.connect_cluster:
        print(cluster_dispatcher(args.connect_cluster))

    with open(SELECTED_TILES_FILE, 'r') as file:
        all_tiles = yaml.load(file, Loader=yaml.FullLoader)
        aoi_tiles = all_tiles.get(args.aoi, None)
    if aoi_tiles is None:
        raise Exception(f"Please provide the selected tiles for {args.aoi} in file {SELECTED_TILES_FILE}.")

    rasterize(aoi_tiles, args.year, args.file_to_raster, args.s3, args.column, args.output_dirs, args.output_formats,
              args.action)


if __name__ == '__main__':
    main()
