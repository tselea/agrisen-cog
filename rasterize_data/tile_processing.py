import argparse
import os

import geopandas as gpd
import pandas as pd
import yaml
from pystac_client import Client
from shapely.geometry import box

STAC_CATALOGUE_URL = 'https://earth-search.aws.element84.com/v0'
AOI_BBOX_FILE = 'rasterize_files/aoi_bbox.yaml'
LPIS_BBOX_FILE = 'rasterize_files/lpis_bounds.gpkg'
AOI_BOUNDARIES_DIR = 'rasterize_files/boundaries/'

import logging

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

log = logging.getLogger("tile_processing")


def get_tiles(stac_url, max_cloud, datetime, bbox, output_file):
    root_catalog = Client.open(stac_url)
    query_params = {'query': {"eo:cloud_cover": {"lte": max_cloud}},
                    'collections': ["sentinel-s2-l2a-cogs"],
                    'datetime': datetime,
                    'bbox': bbox,
                    'max_items': 8000,
                    }
    search_result = root_catalog.search(**query_params)
    nr_items = search_result.matched()
    log.info(f'Found {nr_items} items.')

    tiles_dict = {'tile_name': [], 'original_crs': [], 'geometry': []}
    for item in search_result.items():
        tile_name = item.id.split('_')[1]
        if tile_name in tiles_dict['tile_name']:
            pos = tiles_dict['tile_name'].index(tile_name)
            area_old = tiles_dict['geometry'][pos].area
            area_new = box(*item.bbox).area
            if area_new < area_old:
                continue
            tiles_dict['geometry'][pos] = box(*item.bbox)
        else:
            tiles_dict['tile_name'].append(tile_name)
            tiles_dict['original_crs'].append(item.properties['proj:epsg'])
            tiles_dict['geometry'].append(box(*item.bbox))

    tiles_df = pd.DataFrame(tiles_dict)
    log.info(f'Found {tiles_df.shape[0]} different tiles.')
    tile_gpd = gpd.GeoDataFrame(tiles_df, geometry='geometry', crs='EPSG:4326')
    tile_gpd.to_file(output_file, driver='GPKG')

    tiles_df.drop(columns=['geometry'], inplace=True)
    output_file = output_file.split(".")[0]
    tiles_df.to_csv(f'{output_file}.csv')


def main():
    # Parse user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--stac_url', default=STAC_CATALOGUE_URL, required=False,
                        help=f'The STAC catalogue URL. If not specified, the url {STAC_CATALOGUE_URL} is used.')
    parser.add_argument('--max_cloud', default=30, type=int, required=False,
                        help='The maximum percentage of accepted cloud cover (lte). If not specified, 30% is used.')
    parser.add_argument('--datetime', default='2019-04-01T00:00:00Z/2019-11-01T00:00:00Z', required=False,
                        help='The datetime interval to query tiles. If not specified, the default 2019-04-01T00:00:00Z/2019-11-01T00:00:00Z values is used.')
    parser.add_argument('--bbox_selection', default='aoi', required=False,
                        choices=['aoi', 'lpis', 'boundaries'],
                        help='Where to retrieve the bounding box from. One of [\'aoi\', \'lpis\']')
    parser.add_argument('--aoi', type=str, default=None, required=False,
                        help='Name of the AOI to process. Used for bounding box retrieval.')
    parser.add_argument('--output_file', default=None, required=True,
                        help='The output file path for the results. Only local file supported.')

    args = parser.parse_args()

    if args.bbox_selection == 'aoi' and args.aoi is None:
        raise Exception("The aoi parameter must also be specified, if the bbox_selection is set to aoi.")

    if args.bbox_selection == 'aoi':
        with open(AOI_BBOX_FILE, 'r') as file:
            aoi_bbox = yaml.load(file, Loader=yaml.FullLoader)
            bbox = aoi_bbox.get(args.aoi, None)
    elif args.bbox_selection == 'lpis':
        lpis_gpd = gpd.read_file(LPIS_BBOX_FILE)
        bbox = lpis_gpd.loc[lpis_gpd['AOI'] == args.aoi, 'geometry'].iloc[0].bounds
    if bbox is None:
        raise Exception("The bounding box can not be None.")

    get_tiles(args.stac_url, args.max_cloud, args.datetime, bbox, args.output_file)


if __name__ == '__main__':
    main()
