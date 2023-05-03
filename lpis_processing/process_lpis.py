import argparse
import time
from pathlib import Path

import yaml
from deep_translator import GoogleTranslator

import config_s3 as config
from cluster_connect import cluster_dispatcher
from lpis_processing.utils import get_encoding, create_output_file, read_parquet

key = config.key
secret = config.secret
endpoint_url = config.endpoint_url
bucket_name = config.bucket_name

LPIS_COLUMN_MAPPING_FILE = Path('lpis_files/lpis_column_mapping.yaml')
TRANSLATIONS_DIR = Path('lpis_files/translations')


def generate_translation(input_file, output_file, encoding, is_s3, aoi, year, lang_code):
    lpis_dgpd = read_parquet(input_file, is_s3, key, secret, endpoint_url)

    with open(LPIS_COLUMN_MAPPING_FILE, 'r', encoding='utf-8') as file:
        column_mapping = yaml.load(file, Loader=yaml.FullLoader)

    if aoi not in column_mapping:
        raise Exception(
            f"{aoi} does not have a column mapping. Please specify one of {list(column_mapping.keys())} or add a new column mapping to {LPIS_COLUMN_MAPPING_FILE}")

    if year in column_mapping[aoi]:
        column_mapping = column_mapping[aoi][year]
    else:
        column_mapping = column_mapping[aoi]
    translation_dict = {}
    columns_to_translate = lpis_dgpd.select_dtypes(include='object').columns

    start = time.time()

    for column in column_mapping:
        exists = False
        crops_list = []
        if type(column_mapping[column]) == list:
            for c in column_mapping[column]:
                if c not in columns_to_translate:
                    continue
                crops_list.extend(list(lpis_dgpd[c].unique()))
                exists = True
            crops_list = list(set(crops_list))
        else:
            if column_mapping[column] in columns_to_translate:
                exists = True
                crops_list = list(lpis_dgpd[column_mapping[column]].unique())

        if not exists:
            continue
        translation_dict[column] = {}
        for crop in crops_list:
            if crop:
                try:
                    translation_dict[column][crop] = GoogleTranslator(source=lang_code,
                                                                      target='en').translate(crop).capitalize()
                except:
                    translation_dict[column][crop] = None

    end = time.time()
    print(end - start)
    with open(output_file, 'w') as file:
        yaml.dump(translation_dict, file, allow_unicode=True, encoding='utf-8')


def process(input_file, output_file, encoding, is_s3, aoi, year, no_translate):
    lpis_dgpd = read_parquet(input_file, is_s3, key, secret, endpoint_url)

    with open(LPIS_COLUMN_MAPPING_FILE, 'r', encoding='utf-8') as file:
        column_mapping = yaml.load(file, Loader=yaml.FullLoader)

    if aoi not in column_mapping:
        raise Exception(
            f"{aoi} does not have a column mapping. Please specify one of {list(column_mapping.keys())} or add a new column mapping to {LPIS_COLUMN_MAPPING_FILE}")

    if not no_translate:
        try:
            with open(TRANSLATIONS_DIR / f'translations_{aoi}.yaml', 'r', encoding='utf-8') as file:
                trans_mapping = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise Exception(
                f"{aoi} does not have an lpis file mapping. Add a new lpis translation file mapping to {TRANSLATIONS_DIR}")

    if year in column_mapping[aoi]:
        column_mapping = column_mapping[aoi][year]
    else:
        column_mapping = column_mapping[aoi]

    sel_columns = list(column_mapping.values())
    sel_columns.append('geometry')

    if any(isinstance(n, list) for n in sel_columns):
        l = []
        for i in range(len(sel_columns) - 1, 0, -1):
            if type(sel_columns[i]) == list:
                l.extend(sel_columns[i])
                sel_columns.pop(i)
        sel_columns.extend(l)

    if 'area_m' in column_mapping:
        lpis_dgpd = lpis_dgpd.assign(area_ha=lambda x: x[[column_mapping['area_m']]] / 10000)

        sel_columns.remove(column_mapping['area_m'])
        sel_columns.append('area_ha')
        del column_mapping['area_m']

    # select only the columns of interest
    lpis_dgpd = lpis_dgpd[lpis_dgpd.columns.intersection(sel_columns)]

    trans_columns = {}
    multiple_cols = {}
    for k, v in column_mapping.items():
        if type(v) == list:
            for i, vv in enumerate(v):
                trans_columns[vv] = f'{k}_{i + 1}'
                multiple_cols[f'{k}_{i + 1}'] = k
        else:
            trans_columns[v] = k

    lpis_dgpd = lpis_dgpd.rename(columns=trans_columns)

    start = time.time()

    if not no_translate:
        columns_to_translate = lpis_dgpd.select_dtypes(include='object').columns
        for column in columns_to_translate:
            translation = {}
            if column in trans_mapping:
                translation = trans_mapping.get(column)
            else:
                translation = trans_mapping.get(multiple_cols.get(column))
            lpis_dgpd[column] = lpis_dgpd[column].replace(translation)
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

    parser.add_argument('--input_file', required=True,
                        help='Parquet input file path to translate. Local path or S3')
    parser.add_argument('--output_file', default=None, required=False,
                        help='Parquet translated output file path. Local path or S3')
    parser.add_argument('--s3', action='store_true', default=False, required=False,
                        help='Input/output source Parquet store.')

    parser.add_argument('--encoding', default=None, required=False,
                        help='Encoding to load the file. Default is utf-8.')
    parser.add_argument('--aoi', type=str, default=None, required=True,
                        help='Name of the AOI to process. Used for column mapping and languge. Also used for encoding, if encoding is not specified.')
    parser.add_argument('--year', type=int, default=None, required=False,
                        help='Year of the AOI to process. Required only if there are specific column naming for the same AOI, but different year')

    parser.add_argument('--generate_trans', default=None, required=False,
                        help='File path of the generated lpis translation file mapping.')
    parser.add_argument('--lang_code', default=None, required=False,
                        help='ISO 639-1 language code to make the English translation from.')

    parser.add_argument('--no_translate', action='store_true', default=False, required=False,
                        help='Input/output source Parquet store.')

    args = parser.parse_args()

    if args.connect_cluster:
        print(cluster_dispatcher(args.connect_cluster))

    encoding = get_encoding(args.encoding, args.aoi, args.year)

    output_file = create_output_file(args.output_file, args.aoi)

    if args.generate_trans:
        generate_translation(args.input_file, args.generate_trans, encoding, args.s3, args.aoi, args.year,
                             args.lang_code)
    else:
        process(args.input_file, output_file, encoding, args.s3, args.aoi, args.year, args.no_translate)


if __name__ == '__main__':
    main()
