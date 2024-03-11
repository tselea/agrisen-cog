# LPIS Files Directory Documentation

Detailed documentation for the `lpis_files` directory.

# Table Of Contents
1. [LPIS Columns](#lpis_columns)
2. [LPIS Encodings](#encodings)
3. [LPIS English Translations](#translations)
4. [ICC Code Mappings](#icc_codes)

## LPIS Columns <a name="lpis_columns"></a>

The `lpis_column_mapping.yaml` file defines column mappings for Land Parcel Identification System (LPIS) datasets across different regions or countries. Each entry maps local dataset column names to standardized names used by your scripts, enabling consistent data processing. This file supports dynamic adaptation to various data schemas and accommodates year-specific dataset variations.

### Description Scheme

- **Top-Level Keys**: Represent countries or regions (e.g., Austria, Belgium).
- **Sub-Keys and Values**: Map standardized names (e.g., `crop_type`) to actual dataset column names.
- **Year-Specific Mappings**: Provide mappings that vary by year under the respective country or region.

### Example Entry
```yaml
Netherlands:
  2019:
    crop_type: 'GWS_GEWAS'
    crop_group: 'CAT_GEWASCATEGORIE'
    area_m: 'Shape_Area'
  2020:
    crop_type: 'gewas'
    crop_group: 'category'
```
## LPIS Encodings <a name="encodings"></a>

The `encodings.yaml` file specifies the character encodings for Land Parcel Identification System (LPIS) datasets for various regions or countries and different years. This file ensures that your scripts correctly interpret and process text data from diverse sources, preventing encoding errors and data corruption.

### Description Scheme

- **Top-Level Keys**: Correspond to countries or regions (e.g., Austria, Belgium).
- **Year-Specific Encodings**: Each country or region entry contains sub-keys for years, mapping to the character encoding used for that year's dataset.

### Example Entry

```yaml
Catalonia:
  2018: cp1252
  2019: utf-8
  2020: utf-8

```

This format allows for precise control over text encoding, accommodating variations across different datasets and ensuring seamless data processing workflows.

## LPIS English Translations <a name="translations"></a>

The `translations` directory, contains an English translation file for each AOI, such as `translation_Austria.yaml`, containing the language translation mappings for agricultural crop types within LPIS.

### Description Scheme
- **Column name**: The name of the column to translate
  - **Mappings**: The body of the file lists local crop names as keys and their English translations as values.

### Example Entry

```yaml
crop_type:
  AGROSTIS: Agrostis
  ALBERCOQUERS: Apricots
  ALBERGÍNIA: Eggplant
  ALFALS NO SIE: Alfals
```
This structure ensures that data from different sources can be aggregated or compared accurately by translating diverse local terminologies into a unified language, facilitating analysis and reporting.

## ICC Code Mappings <a name="icc_codes"></a>

The `icc_codes` directory, contains an iCC code mapping file for each AOI, such as `Austria_crops.yaml`, containing the ICC code mappings for agricultural crop types within LPIS.

### Description Scheme
- **Crop name**: The name of the crop 
  - **group**: The corresponding group name as defined in the ICC.
  - **class**: The corresponding class name as defined in the ICC.
  - **code**: The numeric ICC code.
  - **subclass** : The corresponding subclass name as defined in the ICC.
Based on granularity, some of the above might be missing. 

### Example Entry

```yaml
Sorghum:
  class: Sorghum
  code: '14'
  group: Cereals
```

This structure ensures that data is easily mapped onto ICC categories.