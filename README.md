# AgriSen-COG: a Large-Scale Dataset for Crop Detection

# Table Of Contents
1. [Introduction](#introduction)
3. [Dataset](#data)
4. [Getting Started!](#getstarted)
5. [Model Weights](#model_weights)
6. [Acknowledgements](#acknowledgements)

## Introduction <a name="introduction"></a>

Code for AgriSen-COG dataset. This repository is going to be public once the dataset is published.

This repository introduces the [AgriSen-COG Dataset](https://www.mdpi.com/2072-4292/15/12/2980), a multicountry, multitemporal large-scale Sentinel-2 Benchmark Dataset for Crop Detection.

## Dataset Repositories


## Dropbox

Link: https://www.dropbox.com/sh/5bc55skio0o5xd7/AAAQVG3ZmVGFNvPiltQ9Esqma?dl=0 

- `AgriSen-COG/`: Contains the dataset
  - `intermediate_outputs/`: Contains the intermediate outputs of dataset preparation.
    - `output1_1_original_lpis/`: Contains the original LPIS for each AOI: Austria (2019, 2020), Belgium (2019, 2020), Catalonia(2019, 2020), Denmark (2019, 2020), Netherlands (2019, 2020).
    - `output1_2_lpis_gpkg/`: Contains the original LPIS for each AOI: Austria (2019, 2020), Belgium (2019, 2020), Catalonia(2019, 2020), Denmark (2019, 2020), Netherlands (2019, 2020).
`
## Zenodo

Link: https://doi.org/10.5281/zenodo.7892012

- `output1_1_original_lpis.zip`: Contains the original LPIS for each AOI: Austria (2019, 2020), Belgium (2019, 2020), Catalonia(2019, 2020), Denmark (2019, 2020), Netherlands (2019, 2020).
- `output1_2_lpis_gpkg.zip`: Contains the original LPIS for each AOI: Austria (2019, 2020), Belgium (2019, 2020), Catalonia(2019, 2020), Denmark (2019, 2020), Netherlands (2019, 2020).


## Minio S3 Bucket

Endpoint:
* ~~https://s3-3.services.tselea.info.uvt.ro~~ (updated on the 4th of June 2023)
* https://s3-4.services.tselea.info.uvt.ro (updated on the 7th of March 2024)

Bucket name: `agrisen-cog-v1`

Set anonymous access for download.

## If you use our code, please cite:

Selea, Teodora. "AgriSen-COG, a Multicountry, Multitemporal Large-Scale Sentinel-2 Benchmark Dataset for Crop Mapping Using Deep Learning." Remote Sensing 15.12 (2023): 2980.
```
@article{selea2023agrisen,
  title={AgriSen-COG, a Multicountry, Multitemporal Large-Scale Sentinel-2 Benchmark Dataset for Crop Mapping Using Deep Learning},
  author={Selea, Teodora},
  journal={Remote Sensing},
  volume={15},
  number={12},
  pages={2980},
  year={2023},
  publisher={MDPI}
}

```

Selea, T. (2023). AgriSen-COG.LPIS.GT. [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7892012
```
@dataset{selea2023agrisen,
  author       = {Selea, T.},
  title        = {AgriSen-COG.LPIS.GT.},
  year         = {2023},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.7892012},
  url          = {https://doi.org/10.5281/zenodo.7892012},
}

```
