import argparse
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import dask_geopandas
import dask


from lpis_processing import config_s3 as config
from lpis_processing.utils import SingleDispatcher
from lpis_processing.cluster_connect import cluster_dispatcher


key = config.key
secret = config.secret
endpoint_url = config.endpoint_url
bucket_name = config.bucket_name

anomaly_dispatcher = SingleDispatcher()




def load_pred (icc_code, icc_name, input_dir):
    crop_dir = Path(input_dir) / f'models/model_{icc_code}'
    pred_losses_file = crop_dir /f'pred_losses_model_{icc_code}.json'
    if not pred_losses_file.is_file():
        return None
    with open(pred_losses_file) as f:
        pred_losses = json.load(f)
    return pred_losses


def compute_otsu_thresolding(pred_losses):
    total_weight = len(pred_losses)
    least_variance = -1
    least_variance_threshold = -1

    #nbins = 0.1
    nbins = 0.01
    if max(pred_losses)>=2:
        nbins = 0.1
    else:
        nbins = 0.01
    # create an array of all possible threshold values which we want to loop through
    color_thresholds = np.arange(min(pred_losses) + nbins, max(pred_losses) - nbins, nbins)
    pred_losses = np.array(pred_losses)

    # loop through the thresholds to find the one with the least within class variance
    for color_threshold in color_thresholds:
        bg_pixels = pred_losses[pred_losses < color_threshold]
        weight_bg = len(bg_pixels) / total_weight
        variance_bg = np.var(bg_pixels)

        fg_pixels = pred_losses[pred_losses >= color_threshold]
        weight_fg = len(fg_pixels) / total_weight
        variance_fg = np.var(fg_pixels)

        within_class_variance = weight_fg * variance_fg + weight_bg * variance_bg
        if least_variance == -1 or least_variance > within_class_variance:
            least_variance = within_class_variance
            least_variance_threshold = color_threshold
    return least_variance_threshold

def load_df(icc_code, input_dir, aoi, year):
    #load initial df for each class
    data_file = f'{input_dir}/{aoi}/{year}/barycenter_poly_{icc_code}.parquet'

    df = pd.read_parquet(data_file,  storage_options={
            "key": key,
            "secret": secret,
            "client_kwargs": {"endpoint_url": endpoint_url}
        })
    return df
@anomaly_dispatcher.register('threshold')
def run_thresholding(aoi, year, input_dir, output_dir):

    #load icc code
    start = time.time()
    ICC_CODES_FILE = f'../lpis_processing/lpis_files/stats/crop_encodings/{aoi}_{year}_v1_0_crop_encoding.json'


    # get all unique codes list
    with open(ICC_CODES_FILE) as f:
        icc_codes_data = json.load(f)

    thresholding = {}
    problem = []
    for icc_name, icc_code in icc_codes_data.items():
        # if icc_code!=17:
        #     continue
        # load pred loss file
        pred_losses = load_pred(icc_code, icc_name, input_dir[0])
        if not pred_losses:
            problem.append(icc_code)
            continue
        thrs = compute_otsu_thresolding(pred_losses)
        thresholding[icc_code] = thrs
    output_path = Path(input_dir[0]) / f'thresholds_{aoi}_{year}.json'
    with open(output_path, 'w') as f:
        json.dump(thresholding,f, indent=4, sort_keys=True)

    print(f'Problems: {problem}')
    #save selected indexes
    all_df_list = []
    for icc_name, icc_code in icc_codes_data.items():
        if icc_code in problem:
            continue
        print(f'Starting with {icc_code}')
        # load pred loss file
        pred_losses = np.array(load_pred(icc_code, icc_name, input_dir[0]))
        # filter polygon values - to keep
        selected_index = np.where(pred_losses <= thresholding[icc_code])

        # get corresponding polygon indexes
        df = load_df(icc_code, input_dir[1], aoi, year)
        poly_index_values = df.index.values[selected_index]
        print(poly_index_values.shape)

        # create new dataframe
        df_data = {'poly_index': poly_index_values, 'icc_code': np.full(poly_index_values.shape, icc_code),
                   'icc_name': np.full(poly_index_values.shape, icc_name, dtype=object)}

        selected_df = pd.DataFrame(df_data)
        selected_data_file = f'{output_dir[0]}/{aoi}/{year}/selected_poly_{icc_code}.parquet'
        selected_df.to_parquet(selected_data_file, storage_options={
            "key": key,
            "secret": secret,
            "client_kwargs": {"endpoint_url": endpoint_url}
        })
        print(f'Finished with {icc_code}')
        all_df_list.append(selected_df)
    all_df = pd.concat(all_df_list, ignore_index=True)
    all_df_file = f'{output_dir}/{aoi}/{year}/selected_poly_all.parquet'
    all_df.to_parquet(all_df_file, storage_options={
        "key": key,
        "secret": secret,
        "client_kwargs": {"endpoint_url": endpoint_url}
    })

    end = time.time()
    print(end-start)



def main():
    # Parse user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--connect_cluster', type=str, default=None, required=False,
                        choices=['coiled'],
                        help='Connect to dask cluster. One of [ \'coiled\']')

    parser.add_argument('--aoi', type=str, default=None, required=False,
                        help='Name of the AOI to process.')
    parser.add_argument('--year', type=int, default=None, required=False,
                        help='Year of the AOI to process.')

    parser.add_argument('--input_dir', default=None, required=False,nargs='+',
                        help='path to read the input tiles.')

    parser.add_argument('--output_dir', default=None, required=False, nargs='+',
                         help='Path to save the results.')
    parser.add_argument('--action', type=str, default=None, required=True,
                        choices=['threshold', 'create_df'],
                        help='The action to be performed. One of [\'threshold\']')

    args = parser.parse_args()


    if args.connect_cluster:
        client = cluster_dispatcher(args.connect_cluster)
        print(client.dashboard_link)
    else:
        client = None

    anomaly_dispatcher(args.action,args.aoi, args.year,args.input_dir, args.output_dir )


if __name__ == '__main__':
    main()