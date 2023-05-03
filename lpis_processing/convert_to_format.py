import argparse
from pathlib import Path

import dask_geopandas
import geopandas as gpd

import config_s3 as config
from lpis_processing.cluster_connect import cluster_dispatcher
from utils import SingleDispatcher, get_encoding, create_output_file

# S3 access config
key = config.key
secret = config.secret
endpoint_url = config.endpoint_url
bucket_name = config.bucket_name

LPIS_ENCODINGS_FILE = Path('lpis_processing/lpis_files/encodings.yaml')

convert_dispatcher = SingleDispatcher()


@convert_dispatcher.register('parquet')
def convert_to_parquet(input_file, output_file, encoding, is_s3, new_crs):
    lpis_gpd = gpd.read_file(input_file, encoding=encoding)
    lpis_dgpd = dask_geopandas.from_geopandas(lpis_gpd, npartitions=4)
    lpis_dgpd = lpis_dgpd.repartition(partition_size='100MB')
    if new_crs:
        lpis_dgpd = lpis_dgpd.to_crs(new_crs)
    if is_s3:
        lpis_dgpd.to_parquet(output_file, storage_options={
            "key": key,
            "secret": secret,
            "client_kwargs": {"endpoint_url": endpoint_url}
        })
    else:
        lpis_dgpd.to_parquet(output_file)


@convert_dispatcher.register('gpkg')
def convert_to_gpkg(input_file, output_file, encoding, is_s3, new_crs):
    lpis_gpd = gpd.read_file(input_file, encoding=encoding)
    if new_crs:
        lpis_gpd = lpis_gpd.to_crs(new_crs)
    try:
        lpis_gpd.to_file(output_file, driver='GPKG')
    except:
        lpis_gpd[['geometry']].to_file(output_file, driver='GPKG')


def main():
    # Parse user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--connect_cluster', type=str, default=None, required=False,
                        choices=['coiled'],
                        help='Connect to dask cluster. One of [\'coiled\']')
    parser.add_argument('--input_file', required=True,
                        help='File/ Dir path to convert')
    parser.add_argument('--output_file', default=None, required=False,
                        help='File/ Dir path to convert')
    parser.add_argument('--output_format', type=str, required=True,
                        choices=['parquet', 'gpkg'],
                        help='Output format to use. One of [\'parquet\', \'gpkg\']')
    parser.add_argument('--s3', action='store_true', default=False, required=False,
                        help='Save data on S3. Available only with output Parquet format.')
    parser.add_argument('--encoding', default=None, required=False,
                        help='Encoding to load the file. Default is utf-8.')
    parser.add_argument('--aoi', type=str, default=None, required=False,
                        help='Name of the AOI to process. Is is used if the encoding or the output_file argument are missing. Default is utf-8.')
    parser.add_argument('--year', type=int, default=None, required=False,
                        help='Year of the AOI to process. Is is used if the encoding or the output_file argument are missing. Default is utf-8.')
    parser.add_argument('--new_crs', type=str, default=None, required=False,
                        help='The CRS for the outpur results. If none, the original CRS is preserved.')

    args = parser.parse_args()

    if args.connect_cluster:
        print(cluster_dispatcher(args.connect_cluster))

    if args.output_format != 'parquet' and args.s3:
        raise Exception("The S3 option is available only with Parquet output format.")

    encoding = get_encoding(args.encoding, args.aoi, args.year)
    output_file = create_output_file(args.output_file, args.aoi, args.year)

    if not output_file:
        raise Exception("Either output_file, aoi or year parameter must be specified.")

    convert_dispatcher(args.output_format, args.input_file, output_file, encoding, args.s3, args.new_crs)


if __name__ == '__main__':
    main()
