import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Project Paths
# =============================================================================

# preprocessing.py
# ├── src/
# │   └── preprocessing.py
# ├── configs/
# │   ├── center_mapping.xlsx
# │   └── variables.json
#
# SCRIPT_DIR = scripts/
# PROJECT_DIR = project root

SCRIPT_DIR = Path(__file__).resolve().parent # finds the parent folder of the script
PROJECT_DIR = SCRIPT_DIR.parent

CENTER_MAPPING_FILE = PROJECT_DIR / "configs" / "center_mapping.xlsx"
TRANSFORMATION_CONFIG_FILE = PROJECT_DIR / "configs" / "variables.json"
UNIT_CONVERSION_FILE = PROJECT_DIR / "configs" / "conversion_factors.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """
    Parse command line arguments.

    Modes
    -----
    finetune
        Preprocess only the variables required by the pretrained CRRTnet model
    train
        Preprocess every available feature for training a new model.
    """
    parser = argparse.ArgumentParser(description="Parse arguments for pre-processing pipeline.")

    parser.add_argument("--mode", choices=["predict", "finetune", "train"], required=True, help="finetune preprocesses only the required features for the pretrained CRRTnet model. train preprocesses all candidate features for training a model from scratch")
    parser.add_argument("--multi-select-delimiter", default=",", required=True, help="Indicate the delimiter that separates multi-selection answers.")

    return parser.parse_args()

# =============================================================================
# Helper Functions
# =============================================================================

def normalize_text(text, lowercase=True):
    """
    Normalize a single string.

    Returns missing values unchanged.
    """
    if pd.isna(text):
        return text

    text = str(text).strip()

    if lowercase:
        text = text.lower()

    return text

def normalize_series(series, lowercase=True):
    """
    Normalize every string in a pandas Series while preserving NaN.
    """
    return series.apply(
        lambda x: normalize_text(x, lowercase=lowercase)
    )

def get_dataset_config(transformation_config, dataset_name):
    """
    Return the config block for one dataset.
    """
    datasets = transformation_config.get("datasets", [])

    for dataset_cfg in datasets:
        if dataset_cfg.get("dataset") == dataset_name:
            return dataset_cfg

def build_config_lookup(dataset_cfg):
    """
    Build a variable name to configuration lookup.
    """
    variables = dataset_cfg.get("variables", [])

    return {var["name"]: var for var in variables}

def get_calculated_variables(transformation_config, mode):
    """
    Return calculated variables allowed for the current mode. 

    finetune -> mode == "all"
    train -> mode in ["all", "train"]
    """
    calc_cfg = get_dataset_config(transformation_config, "calculated_variables")
    variables = calc_cfg.get("variables")

    if mode in ["finetune", "predict"]:
        allowed_modes = {"all"}
    elif mode == "train":
        allowed_modes = {"all", "train"}
    
    return [
        var for var in variables if var.get("mode") in allowed_modes
    ]

def split_mapping_by_dataset(feature_mapping_df):
    """
    Split the feature mapping into dataset-specific subsets.
    """
    subsets = {}
    for dataset_name in ["patient_information", "static_variables", "longitudinal_variables"]:
        subsets[dataset_name] = feature_mapping_df.loc[
            feature_mapping_df["dataset"] == dataset_name
        ].copy()

    return subsets

def get_base_mapping_info(row):
    """
    Extracts basic information from the center mapping files.
    """
    return {
        "var_name": row["standard_var_name"],
        "source_file": row["source_file"],
        "source_column_name": row["source_column_name"]
    }

def get_time_mapping_info(row):
    """
    Retrieves any information if date is used for the transforming the column.
    """
    info = {}
    if row["dataset"] == "longitudinal_variables":
        info["source_date_column_name"] = row["source_date_column_name"]
    if row["dataset"] == "longitudinal_variables" or row["transformation_type"] == "datetime_conversion":
        info["source_date_format"] = row["source_date_format"]
    if row["transformation_type"] == "unit_conversion":
        info["source_unit"] = row["source_unit"]

    return info   

def read_mapping_info(row):
    """
    Adds all information into a variable information dictionary.
    """
    var_info = {}
    var_info.update(get_base_mapping_info(row))
    var_info.update(get_time_mapping_info(row))
    return var_info

def read_var_file(source_path, columns):
    """
    Reads in the source file and ensures that 
    """
    if not source_path.exists():
        raise FileNotFoundError(
            f"Please check that source file path exists for: {source_path}"
        )
    # Read only the header first to validate columns
    header_df = pd.read_csv(source_path, nrows=0)
    missing_cols = [col for col in columns if col not in header_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in file {source_path} for {missing_cols}"
        )
    return pd.read_csv(source_path, usecols=columns)
# =============================================================================
# Mapping Validation
# =============================================================================
def validate_mapping_df(feature_mapping_df):
    """
    Validates the center mapping file before preprocessing. This checks any configuration errors before any data is processed.
    """

    feature_mapping_df = feature_mapping_df.copy()

    # -------------------------------------------------------------------------
    # Every variable must specify which source file it comes from.
    # -------------------------------------------------------------------------
    missing_source_file = feature_mapping_df[feature_mapping_df["source_file"].isna()]
    if not missing_source_file.empty:
        missing_vars = ", ".join(missing_source_file["standard_var_name"].astype(str).tolist())
        raise ValueError(
            "Missing source_file values detected.\n\n"
            "Please complete the source_file column in 'center_mapping.xlsx'\n\n"
            f"Variables:\n{missing_vars}"
        )
    
    # -------------------------------------------------------------------------
    # Every variable must specify the column name in the source dataset.
    # -------------------------------------------------------------------------
    missing_source_col_name = feature_mapping_df[feature_mapping_df["source_column_name"].isna()]
    if not missing_source_col_name.empty:
        missing_vars = ", ".join(missing_source_col_name["standard_var_name"].astype(str).tolist())
        raise ValueError(
            "Missing source_column_name values detected.\n\n"
            f"Variables:\n{missing_vars}"
        )
    # -------------------------------------------------------------------------
    # Longitudinal variables require a timestamp column.
    # -------------------------------------------------------------------------
    missing_longitudinal_dates = feature_mapping_df[(feature_mapping_df["source_date_column_name"].isna()) &
                                                    (feature_mapping_df["dataset"] == "longitudinal_variables")]
    if not missing_longitudinal_dates.empty:
        missing_date_vars = ", ".join(missing_longitudinal_dates["standard_var_name"].astype(str).tolist())
        raise ValueError(
            "Some longitudinal variables are missing source_date_column_name.\n\n"
            f"Variables:\n{missing_date_vars}"
        )

    # -------------------------------------------------------------------------
    # Validate that datetime columns have a source_date_format.
    # -------------------------------------------------------------------------
    missing_date_format = feature_mapping_df[
        ((feature_mapping_df["transformation_type"] == "datetime conversion") | (feature_mapping_df["dataset"] == "longitudinal_variables")) &
        (feature_mapping_df["source_date_format"].isna())
    ]
    
    if not missing_date_format.empty:
        missing_date_format_vars = ", ".join(missing_date_format["standard_var_name"].astype(str).tolist())
        raise ValueError(
            "Some variables are missing date formats.\n\n"
            f"Variables:\n{missing_date_format_vars}"
        )

    # -------------------------------------------------------------------------
    # Validate units.
    #
    # Variables marked as "unitless" skip this validation.
    # -------------------------------------------------------------------------
    
    feature_mapping_df["supported_source_units"] = feature_mapping_df["supported_source_units"].str.strip()

    features_with_units = feature_mapping_df[(feature_mapping_df["supported_source_units"].notna()) &
                                             (feature_mapping_df["supported_source_units"] != "unitless")].copy()
    
    # Every variable with units must specify the unit used by the center
    missing_source_unit = features_with_units[features_with_units["source_unit"].isna()]
    if not missing_source_unit.empty:
        missing_units_vars = ", ".join(missing_source_unit["standard_var_name"].astype(str).tolist())
        raise ValueError(
            "Missing source_unit values.\n\n"
            f"Variables:\n {missing_units_vars}"
        )
    # Every variable needs to have units accepted by the pipeline.
    features_with_units["accepted_units"] = features_with_units["supported_source_units"].apply(
        lambda x: [normalize_text(unit) for unit in x.split(",")]
    )
    features_with_units["unit_accepted"] = features_with_units.apply(lambda row: normalize_text(row["source_unit"]) in row["accepted_units"], axis=1)
    invalid_units = features_with_units.loc[~features_with_units["unit_accepted"]]
    if not invalid_units.empty:
        raise ValueError(
            "Unsupported units detected.\n\n"
            "Please convert these variables to one of the accepted units before preprocessing.\n\n"
            f"Variables: {invalid_units['standard_var_name'].astype(str).tolist()}"
        )
    
# =============================================================================
# Transformation Functions
# =============================================================================
def datetime_conversion(series, *, mapping_info):
    """Datetime conversion with specified format."""
    date_format = mapping_info.get("source_date_format")
    return pd.to_datetime(series, format=date_format, errors="coerce")

def validate_categorical_options(series, *, column_name, config):
    """Validates all categorical values recorded in a series is within accepted values."""
    allowed_values = config["transformation"]["options"]["accepted_values"]

    series = normalize_series(series, lowercase=False)
    bad_values = set(series.dropna().unique()) - set(allowed_values)

    if bad_values:
        raise ValueError(
            f"Variable: {column_name}\n"
            f"Invalid categorical values found: {sorted(bad_values)}\n"
            f"Accepted values: {allowed_values}"
        )
    return series

def one_hot_encode(df, *, column_name, config):
    """One hot encodes categorical columns."""
    series = validate_categorical_options(df[column_name], column_name=column_name, config=config)
    accepted_values = config["transformation"]["options"]["accepted_values"]

    categorical = pd.Categorical(series, categories=accepted_values)
    dummies = pd.get_dummies(categorical, prefix=column_name, dtype=int)

    df = df.drop(columns=[column_name]).join(dummies)
    return df

def binarize(series, *, column_name, config):
    """Binarize transforms categorical columns based on specified mapping within config file."""
    series = validate_categorical_options(series, column_name=column_name, config=config)
    mapping = config["transformation"]["options"]["binarize_map"]
    return series.map(mapping)

def unit_conversion(df, *, column_name, mapping_info, config, reference):
    """Transforms units to ones accepted by the model using a conversion factor."""
    transformation_info = config.get("transformation", {}).get("options", {})

    source_unit = mapping_info.get("source_unit")
    var_name = mapping_info.get("var_name", column_name)

    target_unit = transformation_info.get("target_unit")
    accepted_units = transformation_info.get("accepted_units", [])
    unit_family = transformation_info.get("unit_family")

    if source_unit not in accepted_units:
        raise ValueError(f"Source unit is not accepted for {var_name}: {source_unit}")

    series = pd.to_numeric(df[column_name], errors="coerce")

    if target_unit == source_unit:
        return series

    if unit_family == "vasopressor":
        weight = pd.to_numeric(df["CrrtInitiationWeight_kg"], errors="coerce")
        
        if target_unit == "mcg/kg/min":
            if source_unit == "mcg/kg/hr":
                return series / 60
            elif source_unit == "mcg/min":
                return series / weight
            elif source_unit == "mcg/hr":
                return series / weight / 60
            
        elif target_unit == "units/min":
            if source_unit == "units/hr":
                return series / 60
        raise ValueError(
            f"Unsupported vasopressor conversion for {var_name}: "
            f"{source_unit} -> {target_unit}"
        )

    row = reference.loc[
        (reference["unit_family"] == unit_family) &
        (reference["source_unit"] == source_unit) &
        (reference["target_unit"] == target_unit)
    ]

    if row.empty:
        raise ValueError(
            f"No conversion rule found for {var_name}:"
            f"{unit_family}, {source_unit} -> {target_unit}"
            )
    
    multiplier = row["conversion_factor"].iloc[0]
    offset = row["offset"].iloc[0]
    
    return (series + offset) * multiplier

def validate_multi_select_options(series, *, column_name, config, delimiter):
    """Validates instances where the responses are multi-select."""
    accepted = set(config["transformation"]["options"]["accepted_values"])

    def parse(x):
        if pd.isna(x):
            return x
            
        values = [item.strip() for item in str(x).split(delimiter) if item.strip()]

        invalid = set(values) - accepted

        if invalid:
            raise ValueError(
                f"{column_name}: invalid values {sorted(invalid)}"
            )

        return values

    return series.apply(parse)

def multi_select_encode(df, *, column_name, config, delimiter):
    """Encodes multi-selection in an One-Hot format."""
    series = validate_multi_select_options(df[column_name], column_name=column_name, config=config, delimiter=delimiter)
    accepted_values = config["transformation"]["options"]["accepted_values"]

    normalized = series.apply(lambda x: delimiter.join(x) if isinstance(x, list) else x)

    dummies = normalized.str.get_dummies(sep=delimiter)

    for value in accepted_values:
        if value not in dummies.columns:
            dummies[value] = 0

    # Keep only the accepted dummies
    dummies = dummies[accepted_values]
    dummies.columns = [
        f"{column_name}_{c}" for c in dummies.columns
    ]

    df = df.drop(columns=[column_name])
    df = df.join(dummies)

    return df

# Further transformations after initial data transformation
def build_rule_mask(series, condition):
    """Apply a single rule to a Series."""

    if "gt" in condition:
        return series > condition["gt"]
    elif "ge" in condition:
        return series >= condition["ge"]
    elif "lt" in condition:
        return series < condition["lt"]
    elif "le" in condition:
        return series <= condition["le"]
    elif "eq" in condition:
        return series == condition["eq"]
    else:
        raise ValueError(f"Unsupported condition: {condition}")

def apply_boolean_combine(df, output_col, inputs, combine_type):
    values = df[inputs].fillna(0)

    if combine_type == "any":
        df[output_col] = values.eq(1).any(axis=1).astype(int)
    elif combine_type == "all":
        df[output_col] = values.eq(1).all(axis=1).astype(int)
    else:
        raise ValueError(f"Unsupported combine_type: {combine_type}")
    return df

def apply_further_transformation(df, column, further_transform_config):
    """Following initial transformation, some values require further transformation either through thresholding or clipping values."""
    df_copy = df.copy()

    for transform_option, transform_info in further_transform_config.items():
        if transform_option == "data_threshold":
            lower_bound = transform_info.get("min")
            upper_bound = transform_info.get("max")
            df_copy = df_copy[
                (df_copy[column] >= lower_bound) &
                (df_copy[column] <= upper_bound)
            ]
        elif transform_option == "data_clipping":
            lower_bound = transform_info.get("min")
            upper_bound = transform_info.get("max")
            df_copy[column] = df_copy[column].clip(lower=lower_bound, upper=upper_bound)
        elif transform_option == "rules":
            for rule in transform_info:
                condition = rule.get("condition")
                action = rule.get("action")

                mask = build_rule_mask(df_copy[column], condition)

                action_type = action.get("type")

                if action_type == "set":
                    df_copy.loc[mask, column] = action.get("value")

    return df_copy

# apply transformation based on type
def apply_transform(df, column, mapping_info, config, reference, delimiter):
    before_cols = set(df.columns)
    transformation = config.get("transformation", {})
    
    if transformation:
        transform_type = transformation.get("transformation_type")

        if transform_type == "one_hot":
            df = one_hot_encode(df, column_name=column, config=config)
        elif transform_type == "multi_select":
            df =  multi_select_encode(
                df, 
                column_name=column, 
                config=config, 
                delimiter=delimiter
                )
        elif transform_type == "datetime_conversion":
            df[column] = datetime_conversion(
                df[column],
                mapping_info=mapping_info,
            )
        elif transform_type == "binarize":
            df[column] = binarize(df[column], column_name=column, config=config)
        elif transform_type == "unit_conversion":
            df[column] = unit_conversion(
                df,
                column_name=column,
                mapping_info=mapping_info,
                config=config,
                reference=reference,
            )
        else:
            raise ValueError(f"Unsupported transformation type: {transform_type}")

    further_transformation = config.get("further_processing")
    if further_transformation:
        df = apply_further_transformation(df, column, further_transformation)

    new_name = config.get("variable_rename")
    if new_name and new_name != column:
        df = df.rename(columns={column: new_name})

    after_cols = set(df.columns)
    output_columns = list(after_cols - before_cols)

    if not output_columns:
        output_columns = [config.get("variable_rename", column)]

    return df, output_columns

def apply_daily_aggregation(df, columns, config):
    if config["data_type"] == "categorical":
        return (
            df.groupby(["PID", "window_start"])[columns]
              .max()
              .reset_index()
        )

    elif config["data_type"] == "numerical":
        aggregations = config["daily_aggregation"]

        agg_dict = {
            f"{col}_{agg}": (col, agg)
            for col in columns
            for agg in aggregations
        }

        result = (
            df.groupby(["PID", "window_start"])
              .agg(**agg_dict)
              .reset_index()
        )

        rename_map = config.get("output_names", {})
        if rename_map:
            result = result.rename(columns=rename_map)

        return result

# =============================================================================
# Calculation Functions
# =============================================================================  

def calc_arithmetic(df, input_cols, output_col, coefficients=None):
    result = pd.Series(0.0, index=df.index)

    if coefficients is None:
        raise ValueError(f"Coefficients does not exist for {output_col}")

    for col in input_cols:
        coef = coefficients.get(col, 1.0)
        result += df[col].fillna(0) * coef

    df[output_col] = result
    return df

def calc_fluid_overload(df, input_cols, output_col):
    """
    Calculate percent fluid overload.

    Formula:
        Fluid Balance / Body Weight * 100
    """
    fluid_balance = df[input_cols[0]]
    weight = df[input_cols[1]]

    df[output_col] = (fluid_balance / weight) * 100
    return df

def reference_sum(longterm_var_dict, input_cols, output_col, reference_start_col, reference_end_col):
    """
    Sum a long-term variable by PID and return a patient-level dataframe.
    Assumes one input column for now.
    """
    if len(input_cols) != 1:
        raise ValueError(f"reference_sum expects exactly one input column, got {len(input_cols)}")
    
    input_col = input_cols[0]

    if input_col not in longterm_var_dict:
        raise KeyError(f"{input_col} not found in longterm_var_dict")

    # retrieve longitudinal dataframe
    df = longterm_var_dict[input_col].copy()

    if "PID" not in df.columns:
        raise ValueError(f"'PID' is missing from longterm dataframe for {input_col}")

    if "window_start" not in df.columns:
        raise ValueError(f"'window_start' is missing from longterm dataframe for {input_col}")

    df = df.loc[
        (df["window_start"] >= df[reference_start_col]) &
        (df["window_start"] < df[reference_end_col])
    ].copy()

    result = (
        df.groupby("PID", as_index=False)[input_col]
        .sum()
        .rename(columns={input_col: output_col})
    )
    return result

def reference_closest(raw_var_dict, input_cols, output_col, reference_time_col):
    if len(input_cols) != 1:
        raise ValueError(f"reference_closest expects exactly one input column, got {len(input_cols)}")

    input_col = input_cols[0]

    if input_col not in raw_var_dict:
        raise KeyError(f"{input_col} not found in raw_var_dict")

    df = raw_var_dict[input_col].copy()

    if "PID" not in df.columns or "datetime" not in df.columns:
        raise ValueError(f"'PID' and 'datetime' are required for {input_col}")

    # compare against the actual reference timestamp
    ref = df[reference_time_col]

    df = df.loc[df["datetime"] < ref].copy()
    if df.empty:
        return pd.DataFrame(columns=["PID", output_col])

    df = df.sort_values(["PID", "datetime"])

    result = (
        df.groupby("PID", as_index=False)
          .tail(1)[["PID", input_col]]
          .rename(columns={input_col: output_col})
          .reset_index(drop=True)
    )
    return result

def calc_sofa(df, input_cols, output_col, output_col_prefix="SOFA"):
    """
    Calculate SOFA subscroes and the total SOFA.
    """

    (
        pao2_col,
        fio2_col,
        mech_vent_col,
        platelets_col,
        gcs_col,
        bilirubin_col,
        map_col,
        dobut_col,
        dopa_col,
        epi_col,
        norepi_col,
        creat_col,
        urine_col,
    ) = input_cols

    out = df.copy()

    # RESPIRATORY (PaO2/FiO2 ratio) --------------------------------------------------
    pao2 = pd.to_numeric(out[pao2_col], errors="coerce")
    fio2 = pd.to_numeric(out[fio2_col], errors="coerce")

    fio2_frac = np.where(fio2 > 1, fio2 / 100.0, fio2)
    pf_ratio = pao2 / fio2_frac

    mech_vent = pd.to_numeric(out[mech_vent_col], errors="coerce")

    resp = pd.Series(np.nan, index=out.index)
    valid_pf = pf_ratio.notna()

    # Not mechanically ventilated / no respiratory support
    not_vent = mech_vent.isna() | (mech_vent == 0)
    resp.loc[valid_pf & not_vent & (pf_ratio < 300)] = 2
    resp.loc[valid_pf & not_vent & (pf_ratio < 400) & (pf_ratio >= 300)] = 1
    resp.loc[valid_pf & not_vent & (pf_ratio >= 400)] = 0

    # On mechanical ventilation / respiratory support
    on_vent = mech_vent == 1
    resp.loc[valid_pf & on_vent & (pf_ratio < 100)] = 4
    resp.loc[valid_pf & on_vent & (pf_ratio < 200) & (pf_ratio >= 100)] = 3
    resp.loc[valid_pf & on_vent & (pf_ratio >= 200)] = 2

    # Coagulation ----------------------------------------------------------------------
    platelets = pd.to_numeric(out[platelets_col], errors="coerce")
    coag = pd.Series(np.nan, index=out.index)

    coag.loc[platelets.notna() & (platelets >= 150)] = 0
    coag.loc[platelets.notna() & (platelets < 150) & (platelets >= 100)] = 1
    coag.loc[platelets.notna() & (platelets < 100) & (platelets >= 50)] = 2
    coag.loc[platelets.notna() & (platelets < 50) & (platelets >= 20)] = 3
    coag.loc[platelets.notna() & (platelets < 20) & (platelets >= 0)] = 4

    # Liver -----------------------------------------------------------------------------
    bilirubin = pd.to_numeric(out[bilirubin_col], errors="coerce")
    liver = pd.Series(np.nan, index=out.index)

    liver.loc[bilirubin.notna() & (bilirubin < 1.2)] = 0
    liver.loc[bilirubin.notna() & (bilirubin >= 1.2) & (bilirubin < 2.0)] = 1
    liver.loc[bilirubin.notna() & (bilirubin >= 2.0) & (bilirubin < 6.0)] = 2
    liver.loc[bilirubin.notna() & (bilirubin >= 6.0) & (bilirubin < 12.0)] = 3
    liver.loc[bilirubin.notna() & (bilirubin >= 12.0)] = 4

    # Cardio ----------------------------------------------------------------------------
    map_mmHg = pd.to_numeric(out[map_col], errors="coerce")
    dobut = pd.to_numeric(out[dobut_col], errors="coerce")
    dopa = pd.to_numeric(out[dopa_col], errors="coerce")
    epi = pd.to_numeric(out[epi_col], errors="coerce")
    norepi = pd.to_numeric(out[norepi_col], errors="coerce")

    cardio = pd.Series(np.nan, index=out.index)

    score4 = (
        ((dopa > 15) & dopa.notna())
        | ((epi > 0.1) & epi.notna())
        | ((norepi > 0.1) & norepi.notna())
    )
    score3 = (
        ((dopa > 5) & (dopa <= 15) & dopa.notna())
        | ((epi > 0) & (epi <= 0.1) & epi.notna())
        | ((norepi > 0) & (norepi <= 0.1) & norepi.notna())
    )
    score2 = (
        ((dopa > 0) & (dopa <= 5) & dopa.notna())
        | ((dobut > 0) & dobut.notna())
    )

    cardio.loc[score4] = 4
    cardio.loc[~score4 & score3] = 3
    cardio.loc[~score4 & ~score3 & score2] = 2
    cardio.loc[~score4 & ~score3 & ~score2 & map_mmHg.notna() & (map_mmHg < 70)] = 1
    cardio.loc[~score4 & ~score3 & ~score2 & map_mmHg.notna() & (map_mmHg >= 70)] = 0

    # CNS ----------------------------------------------------------------------------
    gcs = pd.to_numeric(out[gcs_col], errors="coerce")
    cns = pd.Series(np.nan, index=out.index)

    cns.loc[gcs.notna() & (gcs >= 15)] = 0
    cns.loc[gcs.notna() & (gcs < 15) & (gcs >= 13)] = 1
    cns.loc[gcs.notna() & (gcs < 13) & (gcs >= 10)] = 2
    cns.loc[gcs.notna() & (gcs < 10) & (gcs >= 6)] = 3
    cns.loc[gcs.notna() & (gcs < 6) & (gcs >= 0)] = 4

    # Renal ----------------------------------------------------------------------------
    creat = pd.to_numeric(out[creat_col], errors="coerce")
    urine = pd.to_numeric(out[urine_col], errors="coerce")

    creat_score = pd.Series(np.nan, index=out.index)
    creat_score.loc[creat.notna() & (creat < 1.2)] = 0
    creat_score.loc[creat.notna() & (creat >= 1.2) & (creat < 2.0)] = 1
    creat_score.loc[creat.notna() & (creat >= 2.0) & (creat < 3.5)] = 2
    creat_score.loc[creat.notna() & (creat >= 3.5) & (creat < 5.0)] = 3
    creat_score.loc[creat.notna() & (creat >= 5.0)] = 4

    urine_score = pd.Series(np.nan, index=out.index)
    urine_score.loc[urine.notna() & (urine < 200)] = 4
    urine_score.loc[urine.notna() & (urine >= 200) & (urine < 500)] = 3
    urine_score.loc[urine.notna() & (urine >= 500)] = 0

    renal = pd.concat([creat_score, urine_score], axis=1).max(axis=1, skipna=True)

    # Output ----------------------------------------------------------------------------
    out[f"{output_col_prefix}_Resp"] = resp
    out[f"{output_col_prefix}_Coag"] = coag
    out[f"{output_col_prefix}_Liver"] = liver
    out[f"{output_col_prefix}_Cardio"] = cardio
    out[f"{output_col_prefix}_CNS"] = cns
    out[f"{output_col_prefix}_Renal"] = renal

    out[output_col] = out[
        [
            f"{output_col_prefix}_Resp",
            f"{output_col_prefix}_Coag",
            f"{output_col_prefix}_Liver",
            f"{output_col_prefix}_Cardio",
            f"{output_col_prefix}_CNS",
            f"{output_col_prefix}_Renal",
        ]
    ].sum(axis=1, min_count=6)

    return out

def apply_functions(df, formula_name, input_cols, output_col, longterm_var_dict, config):
    if formula_name == "calc_arithmetic":
        coefficients = config.get("coefficients")
        return calc_arithmetic(df, input_cols, output_col, coefficients)
    elif formula_name == "calc_fluid_overload":
        return calc_fluid_overload(df, input_cols, output_col)
    elif formula_name == "reference_sum":
        reference_start_col = config.get("reference_time_start")
        reference_end_col = config.get("reference_time_end")
        return reference_sum(longterm_var_dict, input_cols, output_col, reference_start_col, reference_end_col)
    elif formula_name == "reference_closest":
        reference_time = config.get("reference_time")
        return reference_closest(longterm_var_dict, input_cols, output_col, reference_time)
    elif formula_name == "calc_sofa":
        result = calc_sofa(df, input_cols, output_col, "SOFA")
        return result

# =============================================================================
# Preprocessing Functions
# =============================================================================

def load_center_mapping(center_mapping_file):
    """
    Load the center mapping spreadsheet.
    """
    if not center_mapping_file.exists():
        raise FileNotFoundError(
            "Unable to locate center_mapping.xlsx.\n"
            "Please verify the project directory structure."
        )

    return pd.read_excel(
        center_mapping_file,
        sheet_name="center_mapping",
    )

def load_transformation_config(transformation_config_file):
    """
    Load the JSON transformation config.
    """

    if not transformation_config_file.exists():
        raise FileNotFoundError(
            "Configuration file not found. Please verify that the original project directory structure is maintained."
        )
    
    with open(transformation_config_file, "r") as f:
        transformation_config = json.load(f)

    return transformation_config

def load_unit_conversion(unit_conversion_file):
    """
    Load in the unit conversion configuration file.
    """

    if not unit_conversion_file.exists():
        raise FileNotFoundError(
            "Unit conversion file not found. Please verify that the original project directory structure is maintained."
        )
    return pd.read_csv(unit_conversion_file)

def build_patient_information(patient_info_df):
    """
    Build the patient-level table with patient_information variables.
    """
    # 1. Get PID Row
    pid_row = patient_info_df[patient_info_df["standard_var_name"]=="PID"].iloc[0]
    pid_info = read_mapping_info(pid_row)
    pid_source_path = Path(pid_info["source_file"])
    pid_source_column = pid_info["source_column_name"]

    # 2. Load PID column
    pid_df = read_var_file(pid_source_path, [pid_source_column])
    pid_df = pid_df.drop_duplicates()
    pid_df = pid_df.rename(columns={pid_source_column: "PID"})

    # 3. Load other patient information columns
    patient_info_rows = patient_info_df[patient_info_df["standard_var_name"] != "PID"].copy()
    all_col_var_info = {}
    column_rename_dict = {}

    for source_path, group in patient_info_rows.groupby("source_file"):
        source_path = Path(source_path)

        # Source columns in this file
        group_columns = dict(zip(group["source_column_name"], group["standard_var_name"]))

        # Read PID plus all columns from this file
        cols = list(group_columns.keys()) + [pid_source_column]
        source_file_df = read_var_file(source_path, cols)
        source_file_df = source_file_df.rename(columns={pid_source_column: "PID"})

        # Merge into PID
        pid_df = pd.merge(pid_df, source_file_df, how='left', on="PID")

        # Rename source columns to standard variable names
        column_rename_dict.update(group_columns)
        
        # Save mapping info for later use if needed
        for _, row in group.iterrows():
            all_col_var_info[row["standard_var_name"]] = read_mapping_info(row)
    
    # Rename all patient-info columns
    pid_df = pid_df.rename(columns=column_rename_dict)
    
    return pid_source_column, pid_df, all_col_var_info

def transform_by_lookup(df, mapping_df, mapping_info_by_var, config_lookup, reference, delimiter=","):
    df = df.copy()

    for _, row in mapping_df.iterrows():
        var_name = row["standard_var_name"]

        if var_name == "PID":
            continue

        mapping_info = mapping_info_by_var.get(var_name)
        config = config_lookup.get(var_name)

        df, _ = apply_transform(df, var_name, mapping_info, config, reference, delimiter)
    return df

def build_standardized_patient_frame(patient_df, n_days=7):
    """
    Expand patient-level data into a patient-day frame for prediction windows.
    """
    df = patient_df.copy()
    # anchoring the start and end dates on the date (removing timestamp)
    df["CrrtStartDate"] = df["CrrtStartDate"].dt.normalize()
    df["CrrtEndDate"] = df["CrrtEndDate"].dt.normalize()

    # Repeat each patient row for each prediction day
    expanded_df = df.loc[df.index.repeat(n_days)].copy()

    # Prediction Day
    expanded_df["pred_day"] = expanded_df.groupby(level=0).cumcount()

    # Relative day offset from CRRT start
    expanded_df["rel_day"] = expanded_df["pred_day"] - 1

    # Window Boundaries
    expanded_df["window_start"] = expanded_df["CrrtStartDate"] + pd.to_timedelta(expanded_df["rel_day"], unit="D")

    expanded_df["window_end"] = expanded_df["CrrtStartDate"] + pd.to_timedelta(expanded_df["rel_day"] + 1, unit="D")

    expanded_df["prediction_time"] = expanded_df["window_end"]

    # filter any instances where window end <= last crrt_date
    expanded_df = (
        expanded_df
        .loc[
            expanded_df["window_end"]
            <= expanded_df["CrrtEndDate"]
        ]
        .copy()
    )
    return expanded_df 

def build_static_variable(standardized_df, static_info_df, pid_source_column):
    """
    Build the static variable dataframe to be added to the standardized dataframe. 
    """
    # 1. Take the standardized_df and create a non-duplicate ID list
    static_var_df = standardized_df[["PID"]].drop_duplicates()

    # 2. Load in static variable columns
    all_col_var_info = {}
    column_rename_dict = {}

    for source_path, group in static_info_df.groupby("source_file"):
        source_path = Path(source_path)

        # Map source column names to standardized variable names
        group_columns = dict(zip(group["source_column_name"], group["standard_var_name"]))

        # Read PID plus all static columns from this file
        cols = list(group_columns.keys()) + [pid_source_column]
        source_file_df = read_var_file(source_path, cols)
        
        # Standardize PID column name
        source_file_df = source_file_df.rename(columns={pid_source_column: "PID"})
        
        # Static variables should be one row per patient
        if source_file_df["PID"].duplicated().any():
            duplicated_pids = source_file_df.loc[
                source_file_df["PID"].duplicated(), "PID"
            ].astype(str).tolist()
            raise ValueError(
                f"Duplicated PID values found in static file: {source_path}\n"
                f"Duplicated PIDs: {duplicated_pids}\n\n"
                "Static variables must have exactly one row per patient."
            )

        # Merge into patient dataframe
        static_var_df = pd.merge(static_var_df, source_file_df, how='left', on="PID")

        # Rename source columns to standard variable names
        column_rename_dict.update(group_columns)
        
        # Save mapping info for later use if needed
        for _, row in group.iterrows():
            all_col_var_info[row["standard_var_name"]] = read_mapping_info(row)
    
    # Rename all patient-info columns
    static_var_df = static_var_df.rename(columns=column_rename_dict)
    # Final safety check
    if static_var_df["PID"].duplicated().any():
        raise ValueError("Static variable dataframe contains duplicated PID values after merging.")
    
    return static_var_df, all_col_var_info

def build_transform_longitudinal_variable(standardized_df, longitudinal_info_df, config_lookup, pid_source_column, reference, delimiter):
    """
    Build the longitudinal variable dataframe to be added to the standardized dataframe.
    """
    # 1. Take the standardized_df and create a list of the ID, pred_day, rel_day, window start, and window end.
    longitudinal_var_df = standardized_df[['PID', 'pred_day', 'rel_day', 'window_start', 'window_end']].copy()
    admit_dates_df = standardized_df[['PID', 'HospitalAdmitDate', 'IcuAdmitDate', 'CrrtStartDate']].drop_duplicates()
    longterm_var_dict = {}
    longterm_raw_var_dict = {}
    
    # 2. Load in the longitudinal variable information
    all_col_var_info = {}

    for _, group in longitudinal_info_df.groupby("source_file"):
        for _, row in group.iterrows():
            all_col_var_info[row["standard_var_name"]] = read_mapping_info(row)

    # 3. For each longitudinal variable, load in the file with the data
    for var, info in all_col_var_info.items():
        source_path = Path(info.get("source_file"))
        source_column_name = info.get("source_column_name")
        source_date_column_name = info.get("source_date_column_name")

        source_file_df = read_var_file(source_path, [pid_source_column, source_column_name, source_date_column_name])
        source_file_df = source_file_df.rename(columns={pid_source_column: "PID", source_column_name: var})

        config = config_lookup.get(var)

        # date transform
        source_file_df["datetime"] = datetime_conversion(source_file_df[source_date_column_name], mapping_info=info)
        source_file_df["window_start"] = source_file_df["datetime"].dt.normalize()
        source_file_subset = source_file_df[["PID", "window_start", var]].copy()
        # keep raw rows for closest-prior lookup
        raw_subset = source_file_df[["PID", "datetime", var]].copy()

        # value transform
        source_feature_transformed, output_columns = apply_transform(
            df=source_file_subset,
            column=var,
            mapping_info=info,
            config=config,
            reference=reference,
            delimiter=delimiter
        )
        # aggregate for daily
        source_feature_aggregated = apply_daily_aggregation(
            df=source_feature_transformed,
            columns=output_columns,
            config=config
        )
        
        # add to standardized dataframe
        longitudinal_var_df = pd.merge(
            longitudinal_var_df, 
            source_feature_aggregated, 
            on=["PID", "window_start"],
            how="left"
            )
        # Features that are stored from the admission to before CRRT initiation
        
        store_long_term = config.get("store_long_term")
        if store_long_term:
            longterm_df = pd.merge(
                source_feature_aggregated.copy(),
                admit_dates_df,
                on="PID",
                how="left"
            )
            longterm_df = longterm_df[
                (longterm_df["window_start"] >= longterm_df["HospitalAdmitDate"]) &
                (longterm_df["window_start"] <= longterm_df["CrrtStartDate"])
            ].copy()
            longterm_var_dict[var] = longterm_df

        store_long_term_raw = config.get("store_long_term_raw")
        if store_long_term_raw:
            raw_subset = pd.merge(raw_subset, admit_dates_df, on="PID")
            longterm_raw_var_dict[var] = raw_subset

    return longitudinal_var_df, longterm_var_dict, longterm_raw_var_dict
    

def build_calculated_variables(standardized_df, config, longterm_var_dict, longterm_raw_var_dict):
    # 1. Take the standardized df and create a list of the ID, pred_day
    calculated_var_df = standardized_df.copy()

    for var, info in config.items():
        # Input Components
        inputs = info.get("inputs")

        # Retrieve transformation configuration
        transformation_config = info.get("transformation")
        transformation_type = transformation_config.get("type")

        # Non-formula variables use PID/pred_day + inputs
        if transformation_type != "formula":
            feature_df = calculated_var_df[["PID", "pred_day"] + inputs].copy()
        else:
            formula_name = transformation_config.get("formula_name")
            # Only build feature_df for formulas that need it
            if formula_name not in {"reference_sum", "reference_closest"}:
                feature_df = calculated_var_df[["PID", "pred_day"] + inputs].copy()

        # Transform based on type and rules
        if transformation_type == "rule":
            transformation_rule = transformation_config.get("rules")
            # Set calculated variable to NaN
            feature_df[var] = np.NaN
            # Retrieve input (only one available for rules)
            input_col = inputs[0]
            for rule in transformation_rule:
                condition = rule.get("condition")
                action = rule.get("action")

                mask = build_rule_mask(feature_df[input_col], condition)

                action_type = action.get("type")
                
                if action_type == "set":
                    feature_df.loc[mask, var] = action.get("value") 
        elif transformation_type == "boolean_combine":
            combine_type = transformation_config.get("combine_type")
            feature_df = apply_boolean_combine(feature_df, var, inputs, combine_type)
        elif transformation_type == "formula":
            if formula_name == "reference_sum":
                longterm_df = apply_functions(
                    df=None,
                    formula_name=formula_name,
                    input_cols=inputs,
                    output_col=var,
                    config=transformation_config,
                    longterm_var_dict=longterm_var_dict
                )
                calculated_var_df = pd.merge(calculated_var_df, longterm_df[['PID', var]], how="left", on="PID")
                continue
            elif formula_name == "reference_closest":
                longterm_df = apply_functions(
                    df=None,
                    formula_name=formula_name,
                    input_cols=inputs,
                    output_col=var,
                    config=transformation_config,
                    longterm_var_dict=longterm_raw_var_dict
                )
                calculated_var_df = pd.merge(calculated_var_df, longterm_df[['PID', var]], how="left", on="PID")
                continue
            else:
                feature_df = apply_functions(
                    df=feature_df,
                    formula_name=formula_name,
                    input_cols=inputs,
                    output_col=var,
                    longterm_var_dict=longterm_var_dict,
                    config=transformation_config
                )
        calculated_var_df = pd.merge(calculated_var_df, feature_df[["PID", "pred_day", var]], how="left", on=["PID", "pred_day"])
    return calculated_var_df


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Parse user arguments
    # -------------------------------------------------------------------------
    arg = parse_args()

    # -------------------------------------------------------------------------
    # Load mapping and transformation config
    # -------------------------------------------------------------------------
    center_mapping_df = load_center_mapping(CENTER_MAPPING_FILE)
    transformation_config = load_transformation_config(TRANSFORMATION_CONFIG_FILE)
    unit_conversion_df = load_unit_conversion(UNIT_CONVERSION_FILE)

    # -------------------------------------------------------------------------
    # Select variables based on preprocessing mode.
    # -------------------------------------------------------------------------
    if arg.mode in ["finetune", "predict"]:
        feature_mapping_df = center_mapping_df[center_mapping_df["required"] == True]
    elif arg.mode == "train":
        feature_mapping_df = center_mapping_df.copy()
    
    # -------------------------------------------------------------------------
    # Validate mapping before reading any patient data.
    # -------------------------------------------------------------------------
    validate_mapping_df(feature_mapping_df)

    # -------------------------------------------------------------------------
    # Split mapping into the three data source groups.
    # -------------------------------------------------------------------------
    mapping_groups = split_mapping_by_dataset(feature_mapping_df)
    patient_info_df = mapping_groups["patient_information"]
    static_info_df = mapping_groups["static_variables"]
    longitudinal_info_df = mapping_groups["longitudinal_variables"]

    # -------------------------------------------------------------------------
    # Build config lookups.
    # -------------------------------------------------------------------------
    patient_info_cfg = get_dataset_config(transformation_config, "patient_information")
    static_cfg = get_dataset_config(transformation_config, "static_variables")
    longitudinal_cfg = get_dataset_config(transformation_config, "longitudinal_variables")
    calc_vars = get_calculated_variables(transformation_config, arg.mode)

    patient_info_lookup = build_config_lookup(patient_info_cfg)
    static_lookup = build_config_lookup(static_cfg)
    longitudinal_lookup = build_config_lookup(longitudinal_cfg)
    calc_lookup = {var["name"]: var for var in calc_vars}
    
    # -------------------------------------------------------------------------
    # Step 1. Initialize the Patient Information
    # -------------------------------------------------------------------------

    # Retrieve patient information data
    PID_SOURCE_COLUMN, patient_df, patient_mapping_info = build_patient_information(patient_info_df)
    # Transform patient information data
    patient_df = transform_by_lookup(
        patient_df,
        patient_info_df,
        patient_mapping_info,
        patient_info_lookup,
        unit_conversion_df
    )
    # Create patient-day-level dataframe for static and longitudinal variables
    standardized_df = build_standardized_patient_frame(patient_df, n_days=7)

    # -------------------------------------------------------------------------
    # Step 2. Transform Static Variables
    # -------------------------------------------------------------------------
    
    # Retrieve static variable information data
    static_var_df, static_mapping_info = build_static_variable(standardized_df, static_info_df, PID_SOURCE_COLUMN)
    # Transform static features
    static_var_transformed_df = transform_by_lookup(
        static_var_df, 
        static_info_df,
        static_mapping_info,
        static_lookup,
        unit_conversion_df,
        arg.multi_select_delimiter
    )
    # Merge with standardized dataframe
    standardized_df = pd.merge(standardized_df, static_var_transformed_df, how='left', on="PID")        

    # -------------------------------------------------------------------------
    # Step 3. Transform Longitudinal Variables
    # -------------------------------------------------------------------------
    longitudinal_var_df, longterm_var_dict, longterm_raw_var_dict = build_transform_longitudinal_variable(
        standardized_df, 
        longitudinal_info_df, 
        longitudinal_lookup,
        PID_SOURCE_COLUMN, 
        unit_conversion_df, 
        arg.multi_select_delimiter
    )
    standardized_df = pd.merge(
        standardized_df, 
        longitudinal_var_df, 
        how="left",
        on=["PID", "pred_day", "rel_day", "window_start", "window_end"]
        )
    
    # -------------------------------------------------------------------------
    # Step 4. Add Calculated Variables
    # -------------------------------------------------------------------------
    standardized_df = build_calculated_variables(
        standardized_df=standardized_df,
        config=calc_lookup,
        longterm_var_dict=longterm_var_dict,
        longterm_raw_var_dict=longterm_raw_var_dict
    )

    # -------------------------------------------------------------------------
    # Step 5. Export Transformed Dataframe
    # -------------------------------------------------------------------------
    standardized_df.to_csv(OUTPUT_DIR/"transformed_features.csv", index=False)

if __name__ == "__main__":
    main()