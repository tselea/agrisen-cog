import argparse
import time
from pathlib import Path

import networkx as nx
import yaml

from lpis_processing import config_s3 as config
from lpis_processing.cluster_connect import cluster_dispatcher
from lpis_processing.utils import SingleDispatcher, read_parquet

key = config.key
secret = config.secret
endpoint_url = config.endpoint_url
bucket_name = config.bucket_name
icc_dispatcher = SingleDispatcher()

ICC_FILE = Path('lpis_files/icc.yaml')
LPIS_ICC_DIR = Path('lpis_files/icc_codes')
MIN_HA = 0.1


@icc_dispatcher.register('add_icc_codes')
def add_icc_codes(input_file, input_dir, output_file, is_s3, aoi, years):
    icc_codes_path = LPIS_ICC_DIR / f'{aoi}_crops.yaml'

    with open(icc_codes_path, 'r', encoding='utf-8') as file:
        icc_mapping = yaml.load(file, Loader=yaml.FullLoader)

    lpis_dgpd = read_parquet(input_file, is_s3, key, secret, endpoint_url)

    start = time.time()

    def get_value(row, level):
        info = icc_mapping.get(row['crop_type'], None)
        if info:
            return info.get(level)
        return None

    lpis_dgpd['icc_group'] = lpis_dgpd.apply(get_value, axis=1, meta=(None, 'object'), args=('group',))
    lpis_dgpd['icc_class'] = lpis_dgpd.apply(get_value, axis=1, meta=(None, 'object'), args=('class',))
    lpis_dgpd['icc_subclass'] = lpis_dgpd.apply(get_value, axis=1, meta=(None, 'object'), args=('subclass',))
    lpis_dgpd['icc_order'] = lpis_dgpd.apply(get_value, axis=1, meta=(None, 'object'), args=('order',))
    lpis_dgpd['icc_code'] = lpis_dgpd.apply(get_value, axis=1, meta=(None, 'object'), args=('code',))
    lpis_dgpd = lpis_dgpd.dropna(subset=['icc_code'])
    lpis_dgpd['icc_code'] = lpis_dgpd['icc_code'].astype(int)

    lpis_dgpd.to_parquet(output_file, storage_options={
        "key": key,
        "secret": secret,
        "client_kwargs": {"endpoint_url": endpoint_url}
    })
    end = time.time()
    print(end - start)


@icc_dispatcher.register('filter_by_ha')
def filter_by_ha(input_file, input_dir, output_file, is_s3, aoi, years):
    lpis_dgpd = read_parquet(input_file, is_s3, key, secret, endpoint_url)

    start = time.time()

    if 'area_ha' not in lpis_dgpd.columns:
        lpis_dgpd = lpis_dgpd.assign(area_ha=lambda x: x['geometry'].area / 10000)

    lpis_dgpd = lpis_dgpd[lpis_dgpd["area_ha"] >= MIN_HA]
    lpis_dgpd.to_parquet(output_file, storage_options={
        "key": key,
        "secret": secret,
        "client_kwargs": {"endpoint_url": endpoint_url}
    })
    end = time.time()
    print(end - start)


def main():
    # Parse user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--connect_cluster', type=str, default=None, required=False,
                        choices=['coiled'],
                        help='Connect to dask cluster. One of [\'coiled\']')
    parser.add_argument('--input_file', required=False,
                        help='Parquet input file path to process. Local path or S3')
    parser.add_argument('--input_dir', required=False,
                        help='Parquet input dir path to process. Local path or S3. The file must follow the naming pattern: AOIyyy.parquet')
    parser.add_argument('--output_file', default=None, required=False,
                        help='The output file path for the results. Only local file supported.')
    parser.add_argument('--s3', action='store_true', default=False, required=False,
                        help='Input source Parquet store.')
    parser.add_argument('--aoi', type=str, default=None, required=False,
                        help='Name of the AOI to process. Used for column mapping and languge. Also used for encoding, if encoding is not specified.')
    parser.add_argument('--year', type=int, default=None, required=False, nargs='+',
                        help='Year/s of the AOI to process.')
    parser.add_argument('--action', type=str, default=None, required=True,
                        choices=['add_icc_codes', 'filter_by_ha'],
                        help='The action to be performed. One of [\'add_icc_codes\', \'filter_by_ha\']')

    args = parser.parse_args()

    if args.connect_cluster:
        print(cluster_dispatcher(args.connect_cluster))

    if not args.input_file and not args.input_dir:
        raise Exception("The input file or the input dir must be specified.")

    if args.input_dir:
        if not args.aoi or not args.year:
            raise Exception("The aoi and year/s must be specified if the input dir argument is provided.")

    icc_dispatcher(args.action, args.input_file, args.input_dir, args.output_file, args.s3, args.aoi, args.year)


if __name__ == '__main__':
    main()
