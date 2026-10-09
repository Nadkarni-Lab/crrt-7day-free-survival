# Training a New XGBoost Model
To train a new XGBoost model, you would have to first feature select and then train based on the selected features.

## Feature Selection: mRMR

The feature selection method used during our development was minimum redundancy–maximum relevance (mRMR), compares XGBoost models with different feature counts, and saves the patient assignments for `train.py`. The code file is `feature_selection.py`.

### Before you start
Install dependencies including `pandas`, `numpy`, `scikit-learn`, `xgboost`, and `mrmr-selection` (`pip install mrmr-selection`).

Complete the preprocessing and rolling features step. More information exists in `preprocessing_instructions.md`.
```bash
python src/preprocessing.py --mode train --multi-select-delimiter ','
python src/rolling_features.py --mode train
```
These steps should create:
- `data/outputs/transformed_features.csv`, containing `PID` and binary patient-level `DAY7_Outcome`.
- `data/outputs/model_input.csv`, containing `id`, `time`, and candidate features.

### Run feature selection

```bash
python src/feature_selection.py --seed 42
```

Optionally specify the feature counts to compare:

```bash
python src/feature_selection.py --ks 10 20 30 40 50 80 100 --seed 42
```

The default candidate counts are `1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150`. 

## What the script does

1. Reads `model_input.csv` and joins `DAY7_Outcome` by patient ID if the outcome is not already present. It runs checks for patient-day uniqueness, binary outcomes, and predictors. 
2. Splits on a patient-level into an 80% training and 20% testing. All rows for one patient stay together.
3. Calculates mRMR ranking on training set.
4. Trains a default XGBoost classifier on the top `K` ranked features, adding `time` separately, and evaluates on the test/held-out set.
6. Saves ranking, comparison metrics, exact patient assignments, and run configuration.

### Output files

All outputs are saved to `data/outputs/feature_selection/` by default.

| File | Contents |
| --- | --- |
| `mrmr_features.csv` | Full ranking with `rank` and `column_name`. First row is highest ranked. |
| `feature_selection_metrics.csv` | One row per evaluated feature count, including accuracy, ROC-AUC, class-specific PR-AUC, precision, recall, and F1. |
| `train_test_split.csv` | Unique patient `id` and `split` (`train` or `test`). Reused without resplitting by `train.py`. |
| `feature_selection_config.json` | Seed, tested feature counts, `time` inclusion, and split-method metadata. |

### Select the number of features

Inspect `feature_selection_metrics.csv` and decide which `n_features` provides the best balance of performance and complexity. **The script does not automatically choose the count.** Pass the selected count to `train.py`, for example `--n-features 80`, which was used for our training set.

By default, `time` is **not ranked by mRMR** and is appended to every model's predictor list. Thus `--n-features 80` in the next stage means 80 ranked predictors **plus** `time` (81 model columns). If `--no-time` was used here, use it in `train.py` as well.

### Other options

- `--input`: model input CSV path.
- `--outcomes`: transformed features CSV path for the outcome merge.
- `--output-dir`: destination for selection artifacts.
- `--seed`: split/model random seed (default `42`).
- `--ks`: space-separated list of positive candidate feature counts.
- `--no-time`: omit the additional `time` predictor.
- `--n-jobs`: XGBoost worker count (default `1`).

## Model Training
After mRMR feature selection, `train.py` trains a **new model from scratch** using a manually selected number of ranked features and Optuna hyperparameter optimization. It does not load a pretrained model; see the fine-tuning workflow for that use case.

### Before you start
Install compatible versions of `numpy`, `pandas`, `scikit-learn`, `xgboost`, `optuna`, `joblib`, `tsfresh`, and `mrmr-selection` (plus other dependencies of preprocessing).

**Ensure preprocessing, rolling features, and feature selection is completed before this step.**

The training script expects these existing files:

| Path | Purpose |
| --- | --- |
| `data/outputs/model_input.csv` | Patient-day predictor table; includes `id` and normally `time`. |
| `data/outputs/transformed_features.csv` | Patient outcome source (`PID`, `DAY7_Outcome`). |
| `data/outputs/feature_selection/mrmr_features.csv` | Ranked features in `column_name` order. |
| `data/outputs/feature_selection/train_test_split.csv` | Patient-level `id`, `split` assignments saved during selection. |
| `data/outputs/feature_selection/feature_selection_config.json` | Optional check for consistent `time` inclusion. |

## Run Optuna and train

For example, to use the top **80 mRMR features**:

```bash
python src/train.py \
  --n-features 80 \
  --n-trials 50 \
  --score-optimization roc_auc \
  --seed 42
```

The number of features is a **user choice**, not inferred automatically from the comparison metrics. The number of trials is also a user choice; determine what is the best number of trials for your use case. By default, `time` is appended separately, so the example trains with 81 model columns (80 mRMR features plus `time`). Use `--no-time` only if feature selection also used `--no-time`.

## Training procedure

1. Load the model input and merge the binary patient-level outcome where necessary.
2. **Reuse the exact saved 80/20 patient assignments** from `feature_selection.py`; it does not split the data again.
3. Select the first `--n-features` entries of `mrmr_features.csv` and append `time`.
4. Within the **training** set of patients, optimize XGBoost hyperparameters using Optuna and **five-fold `StratifiedGroupKFold`**. 
5. Maximize mean cross-validated ROC-AUC (default) (`--score-optimization roc_auc`). 
6. Refit an XGBoost classifier using the best Optuna hyperparameters.
7. Generate predictions on held-out and export the model, parameters, metrics, selected predictors, and Optuna trial history.

### Optuna search space

The code follows the original supplied XGBoost search ranges:

| Hyperparameter | Search range |
| --- | --- |
| `n_estimators` | Integer 100–500 |
| `max_depth` | Integer 3–10 |
| `learning_rate` | Float 0.01–0.2 |
| `subsample` | Float 0.25–1.0 |
| `colsample_bytree` | Float 0.5–1.0 |
| `min_child_weight` | Integer 1–10 |
| `gamma` | Float 0–5 |

The model uses `booster='gbtree'` and `objective='binary:logistic'`. The supplied `train.py` does **not** use early stopping during this Optuna procedure.

## Outputs

The model is saved to `data/models/`:

```text
data/models/best_model_mrmr_<number_of_features>feats_roc_auc.joblib
```

The following are saved to `data/outputs/training/`:

| File | Contents |
| --- | --- |
| `selected_features.csv` | Exact ordered predictor list used to fit the model. |
| `best_params.json` | Best Optuna hyperparameters. |
| `optuna_trials.csv` | Trial history for inspection and reproducibility. |
| `test_metrics.json` | Accuracy, ROC-AUC, class-specific AUPRC, precision, recall, F1, and outcome prevalence. |
| `test_predictions.csv` | `id`, `time`, `y_true`, `pred`, and predicted probability `prob`. |


## Command-line options

- `--n-features` (**required**): number of top mRMR-ranked predictors.
- `--n-trials` (default `50`): Optuna trials.
- `--score-optimization` (default `roc_auc`): `roc_auc` or `auprc`.
- `--n-splits` (default `5`): patient-grouped cross-validation folds **inside Optuna**.
- `--seed` (default `42`): reproducibility seed.
- `--n-jobs` (default `1`): XGBoost worker count.
- `--no-time`: omit `time`, matching the feature-selection setting.
- `--input`, `--outcomes`, `--selection-dir`, `--output-dir`, `--model-dir`: override default paths.