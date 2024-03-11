LPIS Processing Package Documentation

Detailed documentation and examples for the `lpis_processing` package.

# Table Of Contents
1. [LPIS Translation Processing](#lpis_processing)
2. [ICC Mapping Processing](#icc_mapping)

## LPIS Translation Processing <a name="lpis_processing"></a>
Detailed instructions on how to use the `process_lpis.py` script.

### Overview
This script is designed for processing and translating Land Parcel Identification System (LPIS) data. It supports reading input files from local filesystems or S3 buckets, applying column mappings and translations based on specified Area of Interest (AOI) and year, and outputting the processed data to a desired location.

### Features
* Reads LPIS data from local or S3 sources.
* Applies predefined column mappings and translations.
* Supports output to local or S3 destinations.
* Optional integration with Dask clusters for distributed computing.

### Functions
* **generate_translation**: Generates a translation mapping file for specified AOI and year based on the unique values in the specified columns. This function is essential for creating a repository of translated terms which can be reused across multiple datasets.

* **process** : Reads the input LPIS file, applies column mappings and translations using the predefined mappings and translation files, then writes the processed data to the specified output location. This function is the core of the script, facilitating the data preparation phase for further analysis or reporting.

### Basic Usage

Generating Translation Mappings
To generate translation mappings for an AOI:
```bash
python -m lpis_processing.process_lpis --generate_trans <output_translation_file_path> --input_file <input_path> --aoi <AOI> --year <year> --lang_code <code> [options]
 ```

Processing LPIS Data
To run the script for processing LPIS data, navigate to the script's directory and use the following command format:
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

Generating Translation Mappings
Generate translation mappings for Austria, for the year 2019:
```bash
python -m lpis_processing.process_lpis --generate_trans path/to/translations/translations_Austria.yaml --input_file s3://agrisen-cog-v1/LPIS_processing/original_files/Austria_2019_distrib.parquet --aoi Austria --year 2019 --lang_code de
```

Processing LPIS Data
Process LPIS data for Catalonia, for the year 2019, with translations:

```bash
python -m lpis_processing.process_lpis --connect_cluster uvt --input_file s3://agrisen-cog-v1/LPIS_processing/original_files/Catalonia/Catalonia_2019_distrib.parquet/ --output_file s3://agrisen-cog-v1/LPIS_processing/en_files/Catalonia_2019_en_distrib.parquet --s3 --aoi Catalonia --year 2019

```
## ICC Mapping Processing <a name="icc_mapping"></a>
Detailed instructions on how to use the `icc_mapping.py` script.


Detailed instructions on how to use the `icc_mapping.py` script.

### Overview

The `icc_mapping.py` script is designed for adding ICC (International Crop Classification) codes to LPIS (Land Parcel Identification System) data. It supports reading input files from local filesystems or S3 buckets, enriching the data with ICC codes based on specified Area of Interest (AOI) and year, and outputting the processed data to a desired location.

### Features

* Reads LPIS data from local or S3 sources.
* Enriches LPIS data with ICC codes based on crop types.
* Supports output to local or S3 destinations.
* Optional integration with Dask clusters for distributed computing.

### Functions

* **add_icc_codes**: This function enriches LPIS data with ICC codes by mapping crop types to their corresponding ICC categories, groups, classes, subclasses, and orders.

### Basic Usage

To enrich LPIS data with ICC codes:

```bash
python -m lpis_processing.icc_mapping --connect_cluster <cluster_type> --input_file <path_to_input_file> --output_file <path_to_output_file> --s3 --aoi <aoi> --year <year> --action add_icc_codes
```

#### Options
- `--connect_cluster` <cluster_name>: Connect to a Dask cluster. Supported clusters: uvt, coiled (optional).
- `--input_file` <input_path>: Path to the input LPIS file. Supports local and S3 paths (required).
- `--output_file` <output_path>: Path for the processed LPIS file. Supports local and S3 paths (required).
- `--s3`: Flag to indicate that input/output paths are S3 locations (optional).
- `--aoi` <AOI>: Area of Interest. Used to select the appropriate ICC mapping file (required).
- `--year` <year>: Year of the AOI data. Used to select the appropriate ICC mapping file (required).
- `--action` <action>: The action to perform. Currently supports add_icc_codes (required).

#### Example Usage
To add ICC codes to LPIS data for Catalonia, for the year 2019, and process the data using a Dask cluster named coiled:

```bash
python -m lpis_processing.icc_mapping --connect_cluster coiled --input_file s3://agrisen-cog-v1/LPIS_processing/en_files/Catalonia_2019_en_distrib.parquet --output_file s3://agrisen-cog-v1/LPIS_processing/icc_en_files/Catalonia_2019_icc_en_parti.parquet --s3 --aoi Catalonia --year 2019 --action add_icc_codes
```