"""
train.py
--------
Train all three models: Dummy baseline, Linear Regression, Neural Network.
Run from repo root:
    python src/train.py --data data/raw/credit_risk.csv --out outputs/models
"""

import argparse
import os
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor

from src.preprocess import load_and_clean, engineer_features, get_splits


def build_nn(n_features: int) -> tf.keras.Model:
    model = Sequential([
        Dense(128, input_shape=(n_features,), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='linear'),  # regression output — not sigmoid
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_all(data_path: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_clean(data_path)
    X, y = engineer_features(df)
    feature_names = list(X.columns)
    n_features = X.shape[1]

    X_train, X_test, y_train_sc, y_test_sc, scaler_X, scaler_y = get_splits(
        X, y, save_scalers_path=output_dir
    )

    from sklearn.model_selection import train_test_split
    _, _, y_train_raw, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train_raw)
    with open(os.path.join(output_dir, 'dummy.pkl'), 'wb') as f:
        pickle.dump(dummy, f)
    print('Dummy trained')

    lr = LinearRegression()
    lr.fit(X_train, y_train_raw)
    with open(os.path.join(output_dir, 'linear_regression.pkl'), 'wb') as f:
        pickle.dump(lr, f)
    print('Linear Regression trained')

    nn = build_nn(n_features)
    best_path = os.path.join(output_dir, 'best_nn.keras')

    history = nn.fit(
        X_train, y_train_sc,
        epochs=60,
        batch_size=64,
        validation_split=0.15,
        callbacks=[
            EarlyStopping(patience=6, monitor='val_loss', restore_best_weights=True),
            ModelCheckpoint(filepath=best_path, monitor='val_loss', save_best_only=True),
        ],
        verbose=1,
    )
    epochs_run = len(history.history['loss'])
    print(f'Neural Network trained — stopped at epoch {epochs_run}')

    meta = {
        'feature_names': feature_names,
        'n_features': n_features,
        'epochs_run': epochs_run,
        'target': 'loan_amnt',
    }
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/credit_risk.csv')
    parser.add_argument('--out',  default='outputs/models')
    args = parser.parse_args()
    train_all(args.data, args.out)
