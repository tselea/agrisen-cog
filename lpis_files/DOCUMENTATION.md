# LPIS Files Directory Documentation

Detailed documentation for the `lpis_files` directory.

# Table Of Contents
1. [LPIS Columns](#lpis_columns)

## LPIS Columns

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
