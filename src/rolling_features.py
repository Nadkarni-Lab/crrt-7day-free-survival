import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# TSFresh
from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import roll_time_series, impute
from tsfresh.feature_extraction.feature_calculators import set_property
from tsfresh.feature_extraction import feature_calculators as fc

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

FINETUNE_COLUMN_CATEGORY_FILE = PROJECT_DIR / "configs" / "finetune_column_category.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "outputs"
TRANSFORMED_DATA_FILE = OUTPUT_DIR / "transformed_features.csv"
FINETUNE_FINAL_FEATURES_LIST = PROJECT_DIR / "configs" / "finetune_model_final_variables.csv"

SHIFT_DAYS = 1 # DEFAULT 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["predict", "finetune", "train"], required=True, help="finetune preprocesses only the required features for the pretrained CRRTnet model. train preprocesses all candidate features for training a model from scratch")

    return parser.parse_args()

def apply_fill_policy(series, policy, df=None, group_col=None):
    if pd.isna(policy):
        return series

    policy = str(policy).strip().lower()

    if policy in {"zero", "0"}:
        return series.fillna(0)

    if policy in {"fill", "ffill_bfill", "ffill+bfill"}:
        if df is not None and group_col is not None:
            filled = df.groupby(group_col, group_keys=False)[series.name].transform(
                lambda g: g.ffill().bfill()
            )
        else:
            filled = series.ffill().bfill()
        
        if filled.isna().any():
            fallback = series.median()
            if pd.isna(fallback):
                fallback = 0
            filled = filled.fillna(fallback)
        return filled
    
    if policy == "ffill":
        if df is not None and group_col is not None:
            return df.groupby(group_col, group_keys=False)[series.name].transform(
                lambda g: g.ffill()
            )
        return series.ffill()
    
    if policy == "bfill":
        if df is not None and group_col is not None:
            return df.groupby(group_col, group_keys=False)[series.name].transform(
                lambda g: g.bfill()
            )
        return series.bfill()
    return series

def fill_columns_by_policy(df, cols, fill_policy_map, group_col=None):
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        out[col] = apply_fill_policy(
            out[col],
            fill_policy_map.get(col),
            df=out,
            group_col=group_col
        )
    return out

def prepare_dynamic_for_tsfresh(dyn_df, dyn_cols, fill_policy_map):
    df = dyn_df.copy()

    # strict ordering by timeline
    df = df.sort_values(["id", "time"]).reset_index(drop=True)

    # apply each column's configured policy
    df = fill_columns_by_policy(df, dyn_cols, fill_policy_map, group_col="id")

    return df

@set_property("fctype", "simple")
def mode(x):
    s = pd.Series(x)
    return s.mode().iloc[0] if len(s) else np.nan

@set_property("fctype", "simple")
def categorical_entropy(x):
    p = pd.Series(x).value_counts(normalize=True)
    return (-p * np.log(p)).sum()

@set_property("fctype", "simple")
def n_transitions(x):
    s = pd.Series(x)
    return int((s.shift() != s).sum() - 1)

@set_property("fctype", "combiner")
def category_proportions(x, param=None):
    p = pd.Series(x).value_counts(normalize=True)
    return [(f"prop__{k}", float(v)) for k, v in p.items()]


fc.mode = mode
fc.categorical_entropy = categorical_entropy
fc.n_transitions = n_transitions
fc.category_proportions = category_proportions

def roll_and_extract_tsfresh_mixed(
    dyn_wide: pd.DataFrame,
    numeric_cols: list,
    categorical_cols: list,
    lookback_days: int = 1,     # 1 = only the previous day (Day 0 when anchoring Day 1)
    exclude_current: bool = True,
    n_jobs: int = 0
) -> pd.DataFrame:
    """
    dyn_wide: columns = ['id', 'time', <dynamic numeric + dynamic dummy cols>]
    Returns: DataFrame with ['id', 'time', <tsfresh features...>] (raw, not scaled)
    """

    # custom set of features I want to extract
    numeric_fc = {
        "mean": None,
        "median": None,
        "standard_deviation": None,
        "variance": None,
        "maximum": None,
        "minimum": None,
        "mean_change": None,
        "mean_abs_change": None,
        "root_mean_square": None,
    }

    categorical_fc = {
        "mode": None,
        "categorical_entropy": None,
        "n_transitions": None
    }

    # 1) Rolling window on numeric
    rolled_numeric = roll_time_series(
        dyn_wide[["id", "time"] + numeric_cols],
        column_id="id",
        column_sort="time",
        max_timeshift=lookback_days,
        min_timeshift=1 if exclude_current else 0,
        rolling_direction=1,
    )

    feats_numeric = extract_features(
        rolled_numeric,
        column_id="id",
        column_sort="time",
        default_fc_parameters=numeric_fc,
        disable_progressbar=True,
        n_jobs=n_jobs,
    )
    feats_numeric = impute(feats_numeric)

    out_numeric = feats_numeric.reset_index()
    if "level_0" in out_numeric.columns:
       out_numeric = out_numeric.rename(columns={"level_0": "id", "level_1": "time"})
    numeric_tsfresh_cols = [c for c in out_numeric if c not in ["id", "time"]]

    # 2) Rolling window on static
    rolled_categorical = roll_time_series(
        dyn_wide[["id", "time"] + categorical_cols],
        column_id="id",
        column_sort="time",
        max_timeshift=lookback_days,
        min_timeshift=1 if exclude_current else 0,
        rolling_direction=1,
    )

    feats_categorical = extract_features(
        rolled_categorical,
        column_id="id",
        column_sort="time",
        default_fc_parameters=categorical_fc,
        disable_progressbar=True,
        n_jobs=n_jobs,
    )
    feats_categorical = impute(feats_categorical)

    out_categorical = feats_categorical.reset_index()
    if "level_0" in out_categorical.columns: 
        out_categorical = out_categorical.rename(columns={"level_0": "id", "level_1": "time"})
    categorical_tsfresh_cols = [c for c in out_categorical if c not in ["id", "time"]]

    out = pd.merge(out_numeric, out_categorical, on=["id", "time"])
    
    return numeric_tsfresh_cols, categorical_tsfresh_cols, out

def merge_time_with_static(tsfresh_df_scaled, static_scaled):
    """
    tsfresh_df_scaled: ['id','time', <scaled tsfresh feature cols...>]
    static_scaled:     ['id',        <scaled static cols...>]
    Returns:           ['id','time', <tsfresh...>, <static...>]
    """
    return tsfresh_df_scaled.merge(static_scaled, on="id", how="left", validate="many_to_one")

def main():
    arg = parse_args()
    # Load in prepared dataset
    if not TRANSFORMED_DATA_FILE.exists():
        raise FileNotFoundError(
            "Please ensure that the data was transformed before this step. If not, please run preprocessing.py. If completed, please ensure that the file is within the data directory"
        )
    transformed_data = pd.read_csv(TRANSFORMED_DATA_FILE)

    # Load in column categories
    if arg.mode in ["finetune", "predict"]:
        if not FINETUNE_COLUMN_CATEGORY_FILE.exists():
            raise FileNotFoundError(
                "Unable to locate the column_category.csv file. Please verify that the project directory structure maintains the same."
            )
        column_categories = pd.read_csv(FINETUNE_COLUMN_CATEGORY_FILE) # check if it exists
    elif arg.mode == "train":
        pass

    # Fill any missing values based on column_categories
    static_cat_cols = column_categories[column_categories["category"] == "categorical_static"]["column_name"].to_list()
    static_num_cols = column_categories[column_categories["category"] == "numerical_static"]["column_name"].to_list()
    dyn_cat_cols = column_categories[column_categories["category"] == "categorical_dynamic"]["column_name"].to_list()
    dyn_num_cols = column_categories[column_categories["category"] == "numerical_dynamic"]["column_name"].to_list()

    # fill policies for ALL columns
    fill_policies = column_categories.set_index("column_name")["value_fill"].to_dict()

    # Only retrieve the features within the column_categories file
    included_columns = column_categories["column_name"].tolist()

    feature_df_copy = transformed_data[["PID", "pred_day"] + included_columns].copy()
    feature_df_copy = feature_df_copy.rename(columns={'PID': 'id', 'pred_day': 'time'})

    # static fill
    static_df = feature_df_copy[["id"] + static_num_cols + static_cat_cols].drop_duplicates(subset=["id"]).copy()
    static_df = fill_columns_by_policy(static_df, static_num_cols + static_cat_cols, fill_policies)

    # dynamic fill
    dyn_df = feature_df_copy[["id", "time"] + dyn_cat_cols + dyn_num_cols].copy()
    dyn_ready = prepare_dynamic_for_tsfresh(dyn_df, dyn_cat_cols + dyn_num_cols, fill_policies)

    # generate tsfresh
    numeric_tsfresh_cols, categorical_tsfresh_cols, tsfresh_raw= roll_and_extract_tsfresh_mixed(
        dyn_ready,
        numeric_cols=dyn_num_cols,
        categorical_cols=dyn_cat_cols,
        lookback_days=SHIFT_DAYS,
        exclude_current=True,   # predict at Day 1 using up-to-Day 0
        n_jobs=0
    )

    tsfresh_dyn = tsfresh_raw[["id", "time"] + numeric_tsfresh_cols + categorical_tsfresh_cols]

    final_featured = merge_time_with_static(tsfresh_dyn, static_df)

    # export out
    final_featured.to_csv(OUTPUT_DIR / 'final_features_rolled.csv', index=False)

    # read in the final features in the model
    if arg.mode in ["finetune", "predict"]:
        if not FINETUNE_FINAL_FEATURES_LIST.exists():
            raise FileNotFoundError(
                "Unable to locate the finetune_model_final_variables.csv. Please verify that the project directory structure maintains the same."
                )
        final_feature_list = pd.read_csv(FINETUNE_FINAL_FEATURES_LIST)
        final_featured[final_feature_list['column_name'].tolist()].to_csv(OUTPUT_DIR/"model_input.csv", index=False)
    
if __name__ == "__main__":
    main()