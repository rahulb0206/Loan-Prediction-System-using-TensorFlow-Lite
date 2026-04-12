"""
evaluate.py
-----------
Regression evaluation for the loan amount prediction pipeline.
Run from repo root:
    python src/evaluate.py --models outputs/models --data data/raw/credit_risk.csv
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

from src.preprocess import load_and_clean, engineer_features, get_splits


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'Model': name,
        'RMSE':  f'${rmse:,.0f}',
        'MAE':   f'${mae:,.0f}',
        'R²':    f'{r2:.4f}',
        'MAPE':  f'{mape:.1f}%',
    }


def accuracy_brackets(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    errors_pct = np.abs(y_true - y_pred) / y_true * 100
    rows = []
    for t in [10, 20, 30, 50, 75]:
        rows.append({
            'Within': f'±{t}%',
            '% of predictions': f'{(errors_pct <= t).mean()*100:.1f}%',
            'Count': int((errors_pct <= t).sum()),
        })
    return pd.DataFrame(rows)


def plot_actual_vs_predicted(y_true, y_pred, model_name, out_dir):
    sample = np.random.choice(len(y_true), min(2000, len(y_true)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true[sample]/1000, y_pred[sample]/1000, alpha=0.25, s=12, color='#1565C0')
    lim = 37
    ax.plot([0, lim], [0, lim], 'r--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel('Actual Loan Amount ($k)')
    ax.set_ylabel('Predicted Loan Amount ($k)')
    ax.set_title(f'{model_name}', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    plt.tight_layout()
    fname = model_name.lower().replace(' ', '_') + '_actual_vs_pred.png'
    plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()


def run_evaluation(models_dir, data_path, figures_dir='outputs/figures'):
    os.makedirs(figures_dir, exist_ok=True)

    df = load_and_clean(data_path)
    X, y = engineer_features(df)
    X_train, X_test, y_train_sc, y_test_sc, scaler_X, scaler_y = get_splits(X, y)

    with open(os.path.join(models_dir, 'scaler_X.pkl'), 'rb') as f: scaler_X = pickle.load(f)
    with open(os.path.join(models_dir, 'scaler_y.pkl'), 'rb') as f: scaler_y = pickle.load(f)
    with open(os.path.join(models_dir, 'dummy.pkl'), 'rb') as f: dummy = pickle.load(f)
    with open(os.path.join(models_dir, 'linear_regression.pkl'), 'rb') as f: lr = pickle.load(f)
    nn = tf.keras.models.load_model(os.path.join(models_dir, 'best_nn.keras'))

    from sklearn.model_selection import train_test_split
    _, X_test_raw, _, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)
    y_true = y_test_raw.values
    X_test_sc = scaler_X.transform(X_test_raw)

    y_pred_dm = dummy.predict(X_test_sc)
    y_pred_lr = lr.predict(X_test_sc)
    y_pred_sc = nn.predict(X_test_sc, verbose=0).flatten()
    y_pred_nn = scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()

    rows = [
        compute_metrics(y_true, y_pred_dm, 'Dummy (mean)'),
        compute_metrics(y_true, y_pred_lr, 'Linear Regression'),
        compute_metrics(y_true, y_pred_nn, 'Neural Network'),
    ]
    metrics_df = pd.DataFrame(rows)
    print('\n=== Model Comparison ===')
    print(metrics_df.to_string(index=False))

    print('\n=== Neural Network — Prediction Accuracy Brackets ===')
    print(accuracy_brackets(y_true, y_pred_nn).to_string(index=False))

    plot_actual_vs_predicted(y_true, y_pred_nn, 'Neural Network', figures_dir)
    return metrics_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',  default='outputs/models')
    parser.add_argument('--data',    default='data/raw/credit_risk.csv')
    parser.add_argument('--figures', default='outputs/figures')
    args = parser.parse_args()
    run_evaluation(args.models, args.data, args.figures)
