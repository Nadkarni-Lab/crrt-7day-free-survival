import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import StratifiedGroupKFold, ParameterSampler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import randint, loguniform

SCRIPT_DIR = Path(__file__).resolve().parent # finds the parent folder of the script
PROJECT_DIR = SCRIPT_DIR.parent

OUTPUT_DIR = PROJECT_DIR / "data" / "outputs"
MODEL_FOLDER = PROJECT_DIR / "data" / "models"
DATA_PATH = OUTPUT_DIR / "model_input.csv"
TRANSFORMED_FEATURES_PATH = OUTPUT_DIR / "transformed_features.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-model-path", required=True, type=Path, help="Path to base trained mode (.joblib)")
    parser.add_argument("--output-predictions-filename", required=True, type=Path)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--n-iter-sampler", default=50)
    return parser.parse_args()

def clean_booster_for_finetuning(base_model):
    booster = base_model.get_booster().copy()
    booster.set_attr(best_iteration=None)
    booster.set_attr(best_score=None)
    return booster

def make_outer_sgkf_split(
    data,
    feature_cols,
    outcome_col="outcome",
    group_col="id",
    n_splits=5,
    random_state=42,
):
    # Make sure each id has one outcome.
    outcome_counts = data.groupby(group_col)[outcome_col].nunique()
    bad_ids = outcome_counts[outcome_counts > 1]

    if len(bad_ids) > 0:
        raise ValueError(
            f"{len(bad_ids)} ids have conflicting outcomes. "
            f"Fix these before splitting."
        )

    missing = [c for c in feature_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing features. First 20: {missing[:20]}")

    X_all = data.loc[:, feature_cols]
    y_all = data[outcome_col].astype(int)
    groups_all = data[group_col]

    outer_cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    train_idx, test_idx = next(outer_cv.split(X_all, y_all, groups=groups_all))

    train_data = data.iloc[train_idx].copy()
    test_data = data.iloc[test_idx].copy()

    return train_data, test_data, train_idx, test_idx

def main():
    args = parse_args()
    # Load in data
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Please ensure that the model input exists from previous `rolling_features.py` script."
        )
    if not TRANSFORMED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Please ensure that `preprocessing.py` was completed before this step and the file `transformed_features.csv` contains the output."
        )
    if not args.base_model_path.exists():
        raise FileNotFoundError(
            "Please ensure that the trained model is placed within the models folder."
        )
    
    data = pd.read_csv(DATA_PATH)
    # Load outcome and merge onto model input
    outcome = (
        pd.read_csv(TRANSFORMED_FEATURES_PATH)[["PID", "DAY7_Outcome"]]
        .drop_duplicates()
        .rename(columns={"PID": "id"})
    )
    data = data.merge(outcome, how="left", on="id")

    if data["DAY7_Outcome"].isna().any():
        raise ValueError("Some rows are missing DAY7_Outcome after merge.")
    
    # Load base model and get expected feature names
    base_model = joblib.load(args.base_model_path)
    feature_cols = base_model.get_booster().feature_names

    # Split data into train and test based on patient + stratification
    outcome_col = "DAY7_Outcome"
    group_col = "id"

    train_data, test_data, outer_train_idx, outer_test_idx = make_outer_sgkf_split(
        data=data,
        feature_cols=feature_cols,
        outcome_col=outcome_col,
        group_col=group_col,
        n_splits=5,
        random_state=args.seed,
    )

    X_train = train_data.loc[:, feature_cols].copy()
    y_train = train_data[outcome_col].astype(int).copy()
    groups_train = train_data[group_col].copy()

    X_test = test_data.loc[:, feature_cols].copy()
    y_test = test_data[outcome_col].astype(int).copy()

    base_num_trees = base_model.get_booster().num_boosted_rounds()

    param_dist = {
        "n_estimators": randint(200, 800),
        "learning_rate": loguniform(0.005, 0.05),
        "max_depth": randint(1, 4),
        "min_child_weight": randint(10, 80),
        "reg_lambda": loguniform(10.0, 150.0),
        "scale_pos_weight_mult": [0.5, 0.75, 1.0],
    }

    samples = list(
        ParameterSampler(
            param_distributions=param_dist,
            n_iter=args.n_iter_sampler,
            random_state=args.seed
        )
    )

    inner_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=args.seed
    )

    best_result = None
    best_score = -np.inf

    for trial_i, sampled_params in enumerate(samples, start=1):
        fold_auc = []
        fold_ap = []
        fold_brier = []
        fold_best_new_trees = []

        for fold_id, (tr_idx, val_idx) in enumerate(
            inner_cv.split(X_train, y_train, groups=groups_train),
            start=1,
        ):
            X_tr = X_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[tr_idx]
            y_val = y_train.iloc[val_idx]

            params = sampled_params.copy()

            # Recalculate scale_pos_weight inside each training fold.
            spw_mult = params.pop("scale_pos_weight_mult")
            pos = int((y_tr == 1).sum())
            neg = int((y_tr == 0).sum())
            params["scale_pos_weight"] = (neg / max(pos, 1)) * spw_mult

            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                tree_method="hist",
                booster="gbtree",
                random_state=args.seed,
                n_jobs=-1,
                early_stopping_rounds=50,
                **params,
            )

            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
                xgb_model=clean_booster_for_finetuning(base_model),
            )

            val_pred = model.predict_proba(X_val)[:, 1]

            fold_auc.append(roc_auc_score(y_val, val_pred))
            fold_ap.append(average_precision_score(y_val, val_pred))
            fold_brier.append(brier_score_loss(y_val, val_pred))

            # Important:
            # With fine-tuning, best_iteration may be total trees.
            # Convert it to number of NEW trees added after the base model.
            if hasattr(model, "best_iteration") and model.best_iteration is not None:
                total_best_trees = model.best_iteration + 1
            else:
                total_best_trees = model.get_booster().num_boosted_rounds()

            best_new_trees = max(1, total_best_trees - base_num_trees)
            fold_best_new_trees.append(best_new_trees)

        mean_auc = float(np.mean(fold_auc))
        mean_ap = float(np.mean(fold_ap))
        mean_brier = float(np.mean(fold_brier))

        result = {
            "trial": trial_i,
            "params": sampled_params.copy(),
            "mean_auc": mean_auc,
            "std_auc": float(np.std(fold_auc, ddof=1)),
            "se_auc": float(np.std(fold_auc, ddof=1) / np.sqrt(len(fold_auc))),
            "mean_ap": mean_ap,
            "std_ap": float(np.std(fold_ap, ddof=1)),
            "se_ap": float(np.std(fold_ap, ddof=1) / np.sqrt(len(fold_ap))),
            "mean_brier": mean_brier,
            "std_brier": float(np.std(fold_brier, ddof=1)),
            "se_brier": float(np.std(fold_brier, ddof=1) / np.sqrt(len(fold_brier))),
            "median_new_trees": int(np.median(fold_best_new_trees)),
            "fold_auc": fold_auc,
            "fold_ap": fold_ap,
            "fold_brier": fold_brier,
            "fold_best_new_trees": fold_best_new_trees,
        }

        score = mean_auc
        if score > best_score:
            best_score = score
            best_result = result

    final_params = best_result["params"].copy()
    spw_mult = final_params.pop("scale_pos_weight_mult")
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    final_params["scale_pos_weight"] = (neg / max(pos, 1)) * spw_mult
    final_params["n_estimators"] = best_result["median_new_trees"]

    final_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        booster="gbtree",
        random_state=42,
        n_jobs=-1,
        **final_params,
    )

    final_model.fit(
        X_train,
        y_train,
        verbose=False,
        xgb_model=clean_booster_for_finetuning(model),
    )

    # Predict on held-out test split
    final_test_prob = final_model.predict_proba(X_test)[:, 1]
    final_test_pred = (final_test_prob >= 0.5).astype(int)

    predictions = test_data[["id"]].copy()
    predictions["y_true"] = y_test.to_numpy()
    predictions["pred"] = final_test_pred
    predictions["prob"] = final_test_prob

    predictions.to_csv(OUTPUT_DIR / args.output_predictions_filename, index=False)

    joblib.dump(final_model, MODEL_FOLDER / 'finetuned_model.joblib')

if __name__ == "__main__":
    main()