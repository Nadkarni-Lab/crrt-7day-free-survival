# Pre-processing
This directory contains the configuration files and instructions needed to map your center's data into the standardized format required by the preprocessing pipeline.

## Table of Contents

- [Getting Started](#getting-started)
- [Choosing Variables to Map](#choosing-variables-to-map)
  - [Fine-tuning the pretrained CRRTnet model](#fine-tuning-the-pretrained-crrtnet-model)
  - [Training a new model](#training-a-new-model)
- [Data Requirements](#data-requirements)
  - [Units](#units)
  - [Dates](#dates)
  - [Binary Variables](#binary-variables)
  - [Longitudinal Variables](#longitudinal-variables)
  - [Static Variables](#static-variables)
  - [Multi-Selection Variables](#multi-selection-variables)
  - [Diagnosis Variables](#diagnosis-variables)
- [Further Feature Information](#further-feature-information)
  - [Derived Variables](#derived-variables)
  - [Value Clipping and Exclusion](#value-clipping-and-exclusion)
- [Instructions to Pre-Process Data](#instructions-to-pre-process-data)

## Getting Started
Before running the preprocessing pipeline, complete the center-specific mapping file located at: `config/center_mapping.csv`. This file defines how your center's data should be mapped to the standardized format expected by the preprocessing pipeline.

For each variable, specify:
* `source_file` - Path to the source file containing the variable.
* `source_column_name` - Name of the column containing the variable in the source file.
* `source_date_column_name` - Name of the associated date/time column for longitudinal variables.
* `source_unit` - Unit of measurement used for the variable.
* `source_date_format` - Format of the associated date/time column.

## Choosing Variables to Map
The variables you need to map depend on your intended use case.

### Fine-tuning the pretrained CRRTnet model
If you are only fine-tuning the pretrained CRRTnet model, complete **only** the rows where
```text
required = TRUE
```
These variables represent the minimum set of features required for model fine-tuning.

### Training a new model
If you plan to train a new model using all available candidate predictors, complete **all** rows in the mapping file. If you would like to include additional variables that are not currently listed in the mapping file, first preprocess the existing variables using the provided pipeline. Additional variables can. then be incorporated into the preprocessed dataset.

## Data Requirements
### Units
The `source_unit` field **must** match one of the supported units listed for each variable under `supported_source_units` for each variable that has an unit.

If your source data use a different unit, convert the values to one of the supported units before running the preprocessing pipeline.

### Dates
For every date or datetime variable, specify the format of the source data in `source_date_format` using Python `strftime` notation. Refer to the Python [documentation](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes) for supported format codes.

### Binary Variables
Binary variables must use the expected encoding specified in `transformation_options`:
- If the source values are already encoded as 0 and 1, ensure that they correspond to the expected. categories. No additional binarization will be performed. 
- If the source values are stored as text (e.g., `Yes`/`No`), they will be automatically binarized during preprocessing.

### Longitudinal Variables
Each longitudinal observation must include an associated timestamp or date so that observations can be correctly aligned during preprocessing. Specify the column containing this information in `source_date_column_name`.

### Static Variables
Static variables represetning patient characteristics, diagnoses, or clinical events must reflect information available **before CRRT initiation**. Filter your data accordingly and ensure that each row corresponds to a unique patient. 

### Multi-Selection Static and Longitudinal Variables
For variables that allow multiple values/selections, separate the selected values using a consistent delimiter. Specify this delimiter when running the preprocessing script. The default delimiter is `,`.

### Diagnosis Variables
Some variables indicate whether a patient has a specific diagnosis, including comorbidities, priror conditions, and primary or secondary diagnoses. When extracting these variables, refer to `docs/diagnosis_mapping_icd_codes.csv` for the ICD codes used to define each diagnosis. Only diagnoses identified **before CRRT initiation** should be included. 
*NOTE: Currently, diagnosis mappings are provided only for diagnoses included in the final feature set used for external validation.*

## Date Extraction Window
Extract data beginning at:
- ICU admission for the selected ICU encounter
and ending at:
- the end of CRRT day 7 (or day 8)

This ensures all values potentially required by the modeling pipeline are available. 

**NOTE:** Please check the `extraction_instructions` for further clarifications of criterias needed for extracting required features. For fine-tuning, some features only need one of the categorical values to be extracted.

## Further Feature Information
Additional information about the features used in the preprocessing pipeline is available in `docs/feature_information.xlsx`.

### Derived Variables
The `calculated_values` sheet documents variables that are **derived** from other fields rather than extracted directly from the source data. 

Refer to this sheet to determine which variable are computed automatically during preprocessing and which must be mapped from your source data.

### Value Clipping and Exclusion
The `data_quality_assurance` sheet documents features for which values are clipped or excluded during preprocessing, including the applicable thresholds or criteria.

## Instructions to Pre-Process Data
After extracting the required data and completed `configs/center_mapping.xlsx`, run the provided preprocessing pipeline.

The preprocessing pipeline consists of two stages:
1. **Standardize the source data** into the format expected by the model.
2. **Generate rolling features** to create the final dataset ingested by the model.

### 1. Standardize the Source Data
Run the script from the project environment using the appropriate preprocessing mode.

For fine-tuning the pretrained CRRTnet model:
```
python src/preprocessing.py --mode finetune --multi-select-delimiter ","
```
For training a new model using all candidate features:
```
python src/preprocessing.py --mode train --multi-select-delimiter ","
```
If your multi-selection variable use a delimiter other than a comma, replace "," with the delimiter used in your source data.
#### Resolve preprocessing errors
The preprocessing pipeline validates the center mapping and source data before and during preprocessing. If an error occurs, review the error message, correct the source data or `center_mapping.xlsx` and rerun the script.

The pieplien checks for common mapping and data-format issues, including:
- missing source files or source columns
- missing date columns or date formats for longitudinal variables
- missing or unsupported units
- invalid categoriccal values
- invalid multi-selection values
- duplicate patient identifiers in static-variable files
#### Intermediate Output
After preprocessing completes successfully, the output will be written to:
```
data/outputs/transformed_features.csv
```
This file will then be used to input into CRRTnet fine-tuning or model-training pipeline.

### 2. Generate Rolling Features
The standardized data must be converted into rolling feature representation expected by the model.

Run:
```
python src/generate_rolling_features.py --mode finetune
```

The script will read: `data/outputs/transformed_features.csv` and use the predefined feature configuration in: `configs/finetune_column_category.csv` to identify static and dynamic variables and apply the appropriate missing-value handling.

#### Dynamic Features
For numerical dynamic variables, the pipeline calculates: mean, median, standard deviation, variance, maximum, minimum, mean change, mean absolute change, and root mean square.

For categorical dynamic variables, the pipeline calculates: mode, categorical entropy, and number of transitions.

Static variables are not rolled and are merged with rolling dynamice features for each patient-day
#### Final Model Input
The complete rolling feature dataset is written to: `data/outputs/final_features_rolled.csv`

For fine-tuning, the pipeline then selects the features required by the pretrained CRRTnet model according to: `configs/finetune_model_final_variables.csv`. The final model input dataset is written to: `data/outputs/model_input.csv`

Once `model_input.csv` has been generated successfully, preprocessing is complete. Proceed to either the training or fine-tuning step.