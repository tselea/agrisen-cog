from pathlib import Path

import dask_geopandas
import yaml

LPIS_ENCODINGS_FILE = Path('lpis_files/encodings.yaml')


def get_encoding(encd, aoi, year=None):
    encoding = 'utf-8'
    if not encd:
        if aoi:
            with open(LPIS_ENCODINGS_FILE) as file:
                encodings_dict = yaml.load(file, Loader=yaml.FullLoader)
                encoding = encodings_dict.get(aoi, 'utf-8')
                if type(encoding) == dict:
                    if year:
                        encoding = encoding.get(year, 'utf-8')
                    else:
                        encoding = list(encoding.values())[-1]
    else:
        encoding = encd

    return encoding


def create_output_file(out_f, aoi, year=None):
    output_file = out_f
    if not out_f:
        if aoi or year:
            output_filename = output_file.split(".")[0]
            output_file = f'{output_filename}_en.parquet'
    if not output_file:
        raise Exception("Either output_file, aoi or year parameter must be specified.")

    return output_file


def read_parquet(input_file, is_s3, key, secret, endpoint_url):
    if is_s3:
        lpis_dgpd = dask_geopandas.read_parquet(input_file, storage_options={
            "key": key,
            "secret": secret,
            "client_kwargs": {"endpoint_url": endpoint_url}
        })
    else:
        lpis_dgpd = dask_geopandas.read_parquet(input_file)
    return lpis_dgpd


class SingleDispatcher(object):
    """Single Dispatcher implementation"""

    def __init__(self):
        self._registry = {}

    def register(self, file_type, func=None):
        """
        Register dispatch of
        :param file_type:
        :return:
        """
        if func is None:
            return lambda f: self.register(file_type, f)
        self._registry[file_type] = func
        return func

    def __call__(self, file_type, *arg, **kwargs):
        if file_type in self._registry:
            return self._registry[file_type](*arg, **kwargs)
