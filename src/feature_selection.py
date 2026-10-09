import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / 'data' / 'outputs' / 'model_input.csv'
DEFAULT_OUTCOMES = ROOT / 'data' / 'outputs' / 'transformed_features.csv'
DEFAULT_OUTPUT = ROOT / 'data' / 'outputs' / 'feature_selection'
DEFAULT_KS = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90,
              100, 110, 120, 130, 140, 150]
EXCLUDED = {'id', 'PID', 'pred_day', 'time', 'DAY7_Outcome', 'outcome',
            'y_true', 'pred', 'prob', 'prediction_time', 'window_start', 'window_end'}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--outcomes', type=Path, default=DEFAULT_OUTCOMES)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--no-time', action='store_true', help='Do not force time into the model')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--ks', type=int, nargs='+', default=DEFAULT_KS,
                   help='Number of mRMR-ranked features; time is added separately')
    return p.parse_args()

def load_data(input_path, outcome_path):
    """Loads the dataset from model_input.csv in and conducts tests."""
    data = pd.read_csv(input_path)
    # Tests
    if 'id' not in data.columns:
        raise ValueError("model_input.csv must include 'id'; preserve it in rolling_features.py")
    if 'time' in data.columns and data.duplicated(['id', 'time']).any():
        raise ValueError('Duplicate (id, time) rows in model input')
    if 'DAY7_Outcome' not in data.columns:
        outcome = pd.read_csv(outcome_path, usecols=['PID', 'DAY7_Outcome'])
        if outcome.groupby('PID')['DAY7_Outcome'].nunique(dropna=False).gt(1).any():
            raise ValueError('Conflicting outcomes within a patient')
        outcome = outcome.drop_duplicates('PID').rename(columns={'PID': 'id'})
        data = data.merge(outcome, on='id', how='left', validate='many_to_one')
    if data['DAY7_Outcome'].isna().any() or not data['DAY7_Outcome'].isin([0, 1]).all():
        raise ValueError('Every row must have a binary DAY7_Outcome')
    if data.groupby('id')['DAY7_Outcome'].nunique().gt(1).any():
        raise ValueError('Each patient must have one consistent outcome')
    
    features = [c for c in data if c not in EXCLUDED]
    if not features:
        raise ValueError("No required features found")
    
    return data, features

def mrmr_rank(X, y, k):
    try:
        from mrmr import mrmr_classif
    except ImportError as exc:
        raise ImportError('Install mRMR: pip install mrmr-selection') from exc
    return list(mrmr_classif(X=X, y=y, K=k))

def split_data(data, seed=42):
    patient_outcomes = data[['id', 'DAY7_Outcome']].drop_duplicates()
    train_ids, test_ids = train_test_split(
        patient_outcomes['id'], test_size=0.2, random_state=seed,
        stratify=patient_outcomes['DAY7_Outcome'],
    )
    train_data = data.loc[data['id'].isin(train_ids)].copy()
    test_data = data.loc[data['id'].isin(test_ids)].copy()
    return train_data, test_data

def calculate_metrics(y_true, y_pred, y_prob):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'pr_auc_1': average_precision_score(y_true, y_prob),
        'pr_auc_0': average_precision_score(1 - y_true, 1 - y_prob),
        'precision_0': precision[0], 'precision_1': precision[1],
        'recall_0': recall[0], 'recall_1': recall[1],
        'f1_0': f1[0], 'f1_1': f1[1],
    }

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.ks or any(k < 1 for k in args.ks):
        raise ValueError('--ks must contain positive integers')

    data, candidates = load_data(args.input, args.outcomes) # loading in the candidate features that were determined within preprocessing
    train_data, test_data = split_data(data, seed=args.seed)
    if not args.no_time: # if want to date the day from CRRT initiation (necessary if wanting to keep true to time-series modeling)
        if 'time' not in data.columns:
            raise ValueError("'time' missing; pass --no-time only if intentionally omitted")
    time_cols = [] if args.no_time else ['time']

    X_train, X_test = train_data[candidates], test_data[candidates]
    y_train = train_data['DAY7_Outcome'].astype(int)
    y_test = test_data['DAY7_Outcome'].astype(int)

    # Generate the ranking once, on training patients only.
    ranking = mrmr_rank(X_train, y_train, k=len(candidates))
    records = []
    for k in args.ks:
        if k > len(ranking):
            print(f'Skipping k={k}: only {len(ranking)} ranked features available')
            continue
        selected = ranking[:k].copy() + time_cols
        model = XGBClassifier(
            objective='binary:logistic', eval_metric='logloss',
            random_state=args.seed, n_jobs=args.n_jobs,
        )
        model.fit(train_data[selected], y_train)
        y_pred = model.predict(test_data[selected])
        y_prob = model.predict_proba(test_data[selected])[:, 1]
        records.append({
            'n_features': k,
            'n_model_columns': len(selected),
            **calculate_metrics(y_test.to_numpy(), y_pred, y_prob),
        })
        print(f'Completed k={k}')

    if not records:
        raise ValueError('No feature counts could be evaluated')
    pd.DataFrame(records).to_csv(args.output_dir / 'feature_selection_metrics.csv', index=False)
    pd.DataFrame({'rank': range(1, len(ranking) + 1), 'column_name': ranking}).to_csv(
        args.output_dir / 'mrmr_features.csv', index=False
    )
    assignments = pd.concat([
        train_data[['id']].drop_duplicates().assign(split='train'),
        test_data[['id']].drop_duplicates().assign(split='test'),
    ], ignore_index=True)
    assignments.to_csv(args.output_dir / 'train_test_split.csv', index=False)
    (args.output_dir / 'feature_selection_config.json').write_text(json.dumps({
        'seed': args.seed,
        'test_size': 0.2,
        'ks': args.ks,
        'time_always_included': not args.no_time,
        'feature_count_excludes_time': True,
        'ranking_fit_on': 'training split only',
        'comparison': 'single stratified patient-level 80/20 split',
        'selection_partition': 'test (used for feature-count comparison; not an untouched final test set)',
    }, indent=2))
    print(f'Saved results to {args.output_dir}')
    print('Review feature_selection_metrics.csv and choose --n-features for train.py')

if __name__ == '__main__':
    main()
