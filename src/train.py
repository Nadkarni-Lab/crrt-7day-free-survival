import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_val_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / 'data' / 'outputs' / 'model_input.csv'
DEFAULT_OUTCOMES = ROOT / 'data' / 'outputs' / 'transformed_features.csv'
DEFAULT_SELECTION_DIR = ROOT / 'data' / 'outputs' / 'feature_selection'
DEFAULT_OUTPUT_DIR = ROOT / 'data' / 'outputs' / 'training'
DEFAULT_MODEL_DIR = ROOT / 'data' / 'models'


def parse_args():
    parser = argparse.ArgumentParser(description='Train XGBoost using mRMR features and Optuna.')
    parser.add_argument('--n-features', type=int, required=True, help='Top K mRMR features; time is added separately.')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--score-optimization', choices=['roc_auc', 'auprc'], default='roc_auc')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-splits', type=int, default=5, help='Patient-grouped CV folds during Optuna only.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--outcomes', type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument('--selection-dir', type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--model-dir', type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument('--no-time', action='store_true', help='Do not append time; must match feature selection.')
    return parser.parse_args()

def load_data(input_path, outcomes_path):
    data = pd.read_csv(input_path)
    if 'id' not in data.columns:
        raise ValueError("model_input.csv must contain 'id'")
    if 'time' in data and data.duplicated(['id', 'time']).any():
        raise ValueError('Duplicate patient-time rows')
    if 'DAY7_Outcome' not in data.columns:
        outcome = pd.read_csv(outcomes_path, usecols=['PID', 'DAY7_Outcome'])
        if outcome.groupby('PID')['DAY7_Outcome'].nunique(dropna=False).gt(1).any():
            raise ValueError('Conflicting outcomes within a patient')
        outcome = outcome.drop_duplicates('PID').rename(columns={'PID': 'id'})
        data = data.merge(outcome, on='id', how='left', validate='many_to_one')
    if data['DAY7_Outcome'].isna().any() or not data['DAY7_Outcome'].isin([0, 1]).all():
        raise ValueError('Every observation must have a binary DAY7_Outcome')
    if data.groupby('id')['DAY7_Outcome'].nunique().gt(1).any():
        raise ValueError('Patient outcomes must be consistent')
    return data

def load_split(data, path):
    split = pd.read_csv(path)
    if not {'id', 'split'}.issubset(split.columns):
        raise ValueError("Split CSV must contain 'id' and 'split'")
    if split['id'].duplicated().any() or not split['split'].isin(['train', 'test']).all():
        raise ValueError('Split assignments must be unique and train/test only')
    merged = data.merge(split, on='id', how='left', validate='many_to_one')
    if merged['split'].isna().any() or set(split['id']) != set(data['id']):
        raise ValueError('Split file and model input have different patient IDs')
    train = merged.loc[merged['split'] == 'train'].copy()
    test = merged.loc[merged['split'] == 'test'].copy()
    if train.empty or test.empty:
        raise ValueError('Both train and test partitions must contain patients')
    return train, test

def load_features(selection_dir, n_features, include_time):
    ranking = pd.read_csv(selection_dir / 'mrmr_features.csv')
    if 'column_name' not in ranking:
        raise ValueError("mrmr_features.csv must contain 'column_name'")
    names = ranking['column_name'].tolist()
    if len(names) != len(set(names)) or any(pd.isna(names)):
        raise ValueError('Invalid or duplicated feature names in mRMR ranking')
    if n_features < 1 or n_features > len(names):
        raise ValueError(f'--n-features must be between 1 and {len(names)}')
    selected = names[:n_features].copy()
    if include_time and 'time' not in selected:
        selected.append('time')
    return selected

def test_metrics(y_true, y_pred, y_prob):
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'auprc_0': float(average_precision_score(1 - y_true, 1 - y_prob)),
        'auprc_1': float(average_precision_score(y_true, y_prob)),
        'precision_0': float(prec[0]), 'precision_1': float(prec[1]),
        'recall_0': float(rec[0]), 'recall_1': float(rec[1]),
        'f1_0': float(f1[0]), 'f1_1': float(f1[1]),
        'prevalence_0': float(np.mean(y_true == 0)),
        'prevalence_1': float(np.mean(y_true == 1)),
    }

def main():
    args = parse_args()
    if args.n_trials < 1 or args.n_splits < 2:
        raise ValueError('--n-trials must be >= 1 and --n-splits must be >= 2')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    config_path = args.selection_dir / 'feature_selection_config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if config.get('time_always_included') != (not args.no_time):
            raise ValueError('Time inclusion differs from feature_selection.py; check --no-time')

    data = load_data(args.input, args.outcomes)
    train, test = load_split(data, args.selection_dir / 'train_test_split.csv')
    feats = load_features(args.selection_dir, args.n_features, not args.no_time)
    missing = [c for c in feats if c not in data]
    if missing:
        raise ValueError(f'Missing selected columns: {missing}')

    X_train, X_test = train[feats], test[feats]
    y_train = train['DAY7_Outcome'].astype(int)
    y_test = test['DAY7_Outcome'].astype(int)
    groups = train['id']
    if y_train.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError('Both train and test must contain both classes')
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    folds = list(cv.split(X_train, y_train, groups))
    for tr, va in folds:
        if y_train.iloc[tr].nunique() != 2 or y_train.iloc[va].nunique() != 2:
            raise ValueError('A CV fold lacks one class; reduce --n-splits')

    def objective(trial):
        params = {
            'booster': 'gbtree', 'objective': 'binary:logistic',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.25, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'verbosity': 0,
        }
        clf = XGBClassifier(**params, random_state=args.seed, n_jobs=1)
        if args.score_optimization == 'roc_auc':
            return float(cross_val_score(
                clf, X_train, y_train, cv=folds, scoring='roc_auc', n_jobs=1
            ).mean())
        probabilities = cross_val_predict(
            clf, X_train, y_train, cv=folds, method='predict_proba', n_jobs=1
        )[:, 1]
        # AUPRC option uses out-of-fold average precision (AP), a standard PR summary.
        return float(average_precision_score(y_train, probabilities))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    best_params = study.best_params
    final_params = {'booster': 'gbtree', 'objective': 'binary:logistic', 'verbosity': 0, **best_params}
    final_model = XGBClassifier(**final_params, random_state=args.seed, n_jobs=args.n_jobs)
    final_model.fit(X_train, y_train)
    probabilities = final_model.predict_proba(X_test)[:, 1]
    predictions = final_model.predict(X_test)
    metrics = test_metrics(y_test.to_numpy(), predictions, probabilities)

    model_path = args.model_dir / f'best_model_mrmr_{args.n_features}feats_{args.score_optimization}.joblib'
    joblib.dump(final_model, model_path)
    pd.DataFrame({'column_name': feats}).to_csv(args.output_dir / 'selected_features.csv', index=False)
    (args.output_dir / 'best_params.json').write_text(json.dumps(best_params, indent=2))
    (args.output_dir / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
    study.trials_dataframe().to_csv(args.output_dir / 'optuna_trials.csv', index=False)
    pd.DataFrame({
        'id': test['id'].to_numpy(),
        'time': test['time'].to_numpy() if 'time' in test else np.nan,
        'y_true': y_test.to_numpy(),
        'pred': predictions,
        'prob': probabilities,
    }).to_csv(args.output_dir / 'test_predictions.csv', index=False)
    print('Best Optuna parameters:', best_params)
    print('Test metrics:', json.dumps(metrics, indent=2))
    print('Saved model:', model_path)

if __name__ == '__main__':
    main()
