import argparse
import time

import stackstac
import yaml
from pystac_client import Client

from lpis_processing import config_s3 as config
from lpis_processing.cluster_connect import cluster_dispatcher
from lpis_processing.utils import SingleDispatcher

import xarray as xr

key = config.key
secret = config.secret
endpoint_url = config.endpoint_url
bucket_name = config.bucket_name

dtw_prep_dispatcher = SingleDispatcher()

SELECTED_TILES_FILE = 'aoi_files/sample_tiles.yaml'
STAC_CATALOGUE_URL = 'https://earth-search.aws.element84.com/v0'
max_cloud = 10
datetime = "2019-01-01T00:00:00Z/2020-01-01T00:00:00Z"


@dtw_prep_dispatcher.register('ndvi')
def compute_ndvi(aoi_tiles, year, bands, output_dir, prefix, input_dir=None, client=None, no_empty=None, icc_code=None,
                 selected_tile=None):
    print(f'Start NDVI for {output_dir.split("/")[-2]} {year}')
    root_catalog = Client.open(STAC_CATALOGUE_URL)

    start = time.time()

    for tile in aoi_tiles:
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
        s2_stack = stackstac.stack(items, resolution=10, assets=bands, chunksize=(-1, 1, 1024, 1024))
        nir, red = s2_stack.sel(band="B08"), s2_stack.sel(band="B04")
        ndvi = (nir - red) / (nir + red)
        ndvi = ndvi.expand_dims(dim='band', axis=1)
        ndvi = ndvi.assign_coords(
            {'title': 'NDVI', 'common_name': 'ndvi', 'center_wavelength': None, 'full_width_half_max': None})

        del ndvi.coords['proj:shape']
        del ndvi.coords['proj:transform']
        output_file = f'{output_dir}/{prefix}_{year}_{tile}.zarr'
        data_ds = ndvi.to_dataset(name=f'{prefix.lower()}')
        data_ds.to_zarr(output_file,
                        storage_options={
                            "key": key,
                            "secret": secret,
                            "client_kwargs": {"endpoint_url": endpoint_url}
                        })

    end = time.time()
    print(end - start)

@dtw_prep_dispatcher.register('ndvi_masked')
def ndvi_masked(aoi_tiles, year, bands, output_dir, prefix, input_dir, client, no_empty,icc_code, selected_tile):

    print(f'Start NDVI for {output_dir.split("/")[-2]} {year}')

    start = time.time()
    root_catalog = Client.open(STAC_CATALOGUE_URL)
    mask_values = [1., 2., 3., 8., 9., 10., 11.]
    #gen_tiles = ['31TDH', '31TBF','31TCF', '31TCG', '31TDF', '31TDG', '31TBE' ,'31TCH', '31TBG']

    gen_tiles = []
    print(f'Start for {selected_tile}')
    for tile in [selected_tile]: #aoi_tiles
        if tile in gen_tiles:
            continue
        if tile!= selected_tile:
            continue
        tile_file = f'{input_dir}/{prefix}_{year}_{tile}.zarr'
        #get mask
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
        s2_stack = stackstac.stack(items, resolution=10, assets=['SCL'], chunksize=(-1, 1, 1024, 1024))

        #load ndvi
        ndvi_xr = xr.open_zarr(tile_file,consolidated=False,
                       storage_options={
                           "key": key,
                           "secret": secret,
                           "client_kwargs": {"endpoint_url": endpoint_url}
                       })['ndvi']

        ndvi_masked_all = []
        for t in range(ndvi_xr.time.data.shape[0]):
            cloud_mask = s2_stack.isel(time=t)
            ndvi_data = ndvi_xr.isel(time=t)
            if str(cloud_mask.coords['time'].data)!= str(ndvi_data.coords['time'].data):
                continue
            ndvi_masked = ndvi_data.where(cloud_mask.isin(mask_values) == False)
            if 'created' not in ndvi_masked.coords:
                ndvi_masked.coords['created'] = str(ndvi_masked.coords['time'])
            ndvi_masked_all.append(ndvi_masked)

        ndvi_masked_xr = xr.concat(ndvi_masked_all, dim='time')
        ndvi_masked_xr2 = ndvi_masked_xr.transpose('time', 'band', 'y', 'x')
        ndvi_masked_xr3 = ndvi_masked_xr2.chunk(dict(time=-1, band=1, y=1024, x=1024))

        print(tile, s2_stack.shape, ndvi_xr.shape, ndvi_masked_xr3.shape)


        output_file = f'{output_dir}/{prefix}_masked_{year}_{tile}.zarr'
        prefix = f'{prefix}_masked'
        del ndvi_masked_xr3.coords['center_wavelength']
        del ndvi_masked_xr3.coords['full_width_half_max']
        del ndvi_masked_xr3.coords['proj:shape']
        del ndvi_masked_xr3.coords['proj:transform']

        data_ds = ndvi_masked_xr3.to_dataset(name=f'{prefix.lower()}')
        data_ds.to_zarr(output_file,
                        storage_options={
                            "key": key,
                            "secret": secret,
                            "client_kwargs": {"endpoint_url": endpoint_url}
                        })
        print(f'Masked tile {tile}')
        gen_tiles.append(tile)
        print(gen_tiles)


    end = time.time()
    print(end - start)


def main():
    # Parse user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--connect_cluster', type=str, default=None, required=False,
                        choices=['coiled'],
                        help='Connect to dask cluster. One of [\'coiled\']')
    parser.add_argument('--aoi', type=str, default=None, required=False,
                        help='Name of the AOI to process.')
    parser.add_argument('--year', type=int, default=None, required=False,
                        help='Year of the AOI to process.')
    parser.add_argument('--bands', default=[], required=False, nargs='+',
                        help='The band names to retrieve')
    parser.add_argument('--output_dir', default=None, required=None,
                        help='S3 path to write the computed result ')
    parser.add_argument('--prefix', default=None, required=True,
                        help='Naming prefix medians for each tile.')
    parser.add_argument('--no_empty', action='store_true', required=False, default=False,
                        help='Eliminate empty patches')
    parser.add_argument('--icc_code', default=None, required=False,
                        help='ICC Code to filter')

    parser.add_argument('--input_dir', default=None, required=False,
                        help='S3 path to read the input tiles.')
    parser.add_argument('--selected_tile', default=None, required=False,
                        help='ndvi masked')

    parser.add_argument('--action', type=str, default=None, required=True,
                        choices=['ndvi', 'ndvi_masked', 'generate_ts', 'get_class_ts', 'barycenter', 'zonal_stats'],
                        help='The action to be performed. One of [\'ndvi\', \'generate_ts\', \'get_class_ts\', \'barycenter\',  \'zonal_stats\']')

    args = parser.parse_args()

    if args.connect_cluster:
        client = cluster_dispatcher(args.connect_cluster)
        print(client.dashboard_link)
    else:
        client = None

    with open(SELECTED_TILES_FILE, 'r') as file:
        all_tiles = yaml.load(file, Loader=yaml.FullLoader)
        aoi_tiles = all_tiles.get(args.aoi, None)
    if aoi_tiles is None:
        raise Exception(f"Please provide the selected tiles for {args.aoi} in file {SELECTED_TILES_FILE}.")

    dtw_prep_dispatcher(args.action, aoi_tiles, args.year, args.bands, args.output_dir, args.prefix, args.input_dir,
                        client, args.no_empty, args.icc_code, args.selected_tile)


if __name__ == '__main__':
    main()
