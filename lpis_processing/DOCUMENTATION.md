LPIS Processing Package Documentation

Detailed documentation and examples for the `lpis_processing` package.

# Table Of Contents
1. [LPIS Translation Processing](#lpis_processing)

## LPIS Translation Processing <a name="lpis_processing"></a>
Detailed instructions on how to use the `process_lpis.py` script.

### Overview
This script is designed for processing and translating Land Parcel Identification System (LPIS) data. It supports reading input files from local filesystems or S3 buckets, applying column mappings and translations based on specified Area of Interest (AOI) and year, and outputting the processed data to a desired location.

### Features
Reads LPIS data from local or S3 sources.
Applies predefined column mappings and translations.
Supports output to local or S3 destinations.
Optional integration with Dask clusters for distributed computing.

### Basic Usage
To run the script, navigate to the script's directory and use the following command format:

```bash
python -m lpis_processing.process_lpis --input_file <input_path> --output_file <output_path> --aoi <AOI> --year <year> [options]

```
### Options

- `--connect_cluster <cluster_name>`: Connect to a Dask cluster. Supported clusters: `uvt`, `coiled` (optional).
- `--input_file <input_path>`: Path to the input LPIS file. Supports local and S3 paths (required).
- `--output_file <output_path>`: Path for the processed LPIS file. Supports local and S3 paths (optional; if not provided, a default will be used based on AOI).
- `--s3`: Flag to indicate that input/output paths are S3 locations (optional).
- `--aoi <AOI>`: Area of Interest. Used for column mapping and language translation (required).
- `--year <year>`: Year of the AOI data. Required for specific column naming (optional).
- `--generate_trans <file_path>`: Generate a translation mapping file for the specified AOI (optional).
- `--lang_code <code>`: ISO 639-1 language code for translation source language (optional).
- `--no_translate`: Skip the translation step (optional).


### Example Usage
```bash
python -m lpis_processing.process_lpis --connect_cluster uvt --input_file s3://phd-ts/AgriSen-COG/original_lpis/Catalonia/Catalonia2019.parquet/ --output_file s3://phd-ts/AgriSen-COG/en_lpis/Catalonia/Catalonia2019.parquet/ --s3 --aoi Catalonia --year 2019

```

