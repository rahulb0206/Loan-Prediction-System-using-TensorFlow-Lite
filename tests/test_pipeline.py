"""
test_pipeline.py
----------------
Sanity checks for the data pipeline. Run from repo root:
    python -m pytest tests/ -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_and_clean, engineer_features, get_splits


DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "credit_risk.csv"
)

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "models"
)


@pytest.fixture(scope="module")
def raw_df():
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def cleaned_df():
    return load_and_clean(DATA_PATH)


@pytest.fixture(scope="module")
def features_and_target(cleaned_df):
    return engineer_features(cleaned_df)


class TestDataLoading:
    def test_file_exists(self):
        assert os.path.exists(DATA_PATH)

    def test_expected_shape(self, raw_df):
        assert raw_df.shape == (32581, 12)

    def test_target_column_exists(self, raw_df):
        assert "loan_amnt" in raw_df.columns

    def test_target_is_continuous(self, raw_df):
        assert raw_df["loan_amnt"].nunique() > 100

    def test_target_range(self, raw_df):
        assert raw_df["loan_amnt"].min() >= 500
        assert raw_df["loan_amnt"].max() <= 35000


class TestCleaning:
    def test_no_nulls_after_cleaning(self, cleaned_df):
        assert cleaned_df.isnull().sum().sum() == 0

    def test_no_rows_dropped(self, raw_df, cleaned_df):
        assert len(cleaned_df) == len(raw_df)

    def test_emp_length_filled(self, cleaned_df):
        assert cleaned_df["person_emp_length"].isnull().sum() == 0

    def test_int_rate_filled(self, cleaned_df):
        assert cleaned_df["loan_int_rate"].isnull().sum() == 0

    def test_int_rate_not_dropped(self, cleaned_df):
        # original code used dropna(axis=1) which deleted this column
        assert "loan_int_rate" in cleaned_df.columns

    def test_emp_length_not_dropped(self, cleaned_df):
        assert "person_emp_length" in cleaned_df.columns

    def test_int_rate_within_expected_range(self, cleaned_df):
        assert cleaned_df["loan_int_rate"].min() >= 5.0
        assert cleaned_df["loan_int_rate"].max() <= 25.0


class TestFeatureEngineering:
    def test_loan_amnt_not_in_features(self, features_and_target):
        X, y = features_and_target
        assert "loan_amnt" not in X.columns

    def test_leaky_feature_dropped(self, features_and_target):
        # loan_percent_income = loan_amnt / person_income — pure target leakage
        X, y = features_and_target
        assert "loan_percent_income" not in X.columns

    def test_loan_status_dropped(self, features_and_target):
        X, y = features_and_target
        assert "loan_status" not in X.columns

    def test_target_shape(self, features_and_target, cleaned_df):
        X, y = features_and_target
        assert len(y) == len(cleaned_df)

    def test_target_is_continuous_regression(self, features_and_target):
        X, y = features_and_target
        assert y.nunique() > 100

    def test_more_features_after_encoding(self, features_and_target):
        X, y = features_and_target
        assert X.shape[1] > 11

    def test_no_raw_categoricals_remain(self, features_and_target):
        X, y = features_and_target
        assert len(X.select_dtypes(include=["object"]).columns) == 0


class TestSplitting:
    def test_split_sizes(self, features_and_target):
        X, y = features_and_target
        X_train, X_test, y_train, y_test, sX, sy = get_splits(X, y, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(X)
        assert abs(len(X_test) / total - 0.2) < 0.01

    def test_train_test_mean_loan_amount_similar(self, features_and_target):
        X, y = features_and_target
        X_train, X_test, y_train, y_test, sX, sy = get_splits(X, y, test_size=0.2)
        assert abs(y_train.mean()) < 0.1
        assert abs(y_test.mean()) < 0.5

    def test_scaler_x_centers_features(self, features_and_target):
        X, y = features_and_target
        X_train, X_test, y_train, y_test, sX, sy = get_splits(X, y, test_size=0.2)
        assert np.abs(X_train.mean(axis=0)).max() < 0.1

    def test_scaler_y_centers_target(self, features_and_target):
        X, y = features_and_target
        X_train, X_test, y_train, y_test, sX, sy = get_splits(X, y, test_size=0.2)
        assert abs(y_train.mean()) < 0.05
        assert abs(y_train.std() - 1.0) < 0.05

    def test_no_test_leakage(self, features_and_target):
        X, y = features_and_target
        X_train, X_test, y_train, y_test, sX, sy = get_splits(X, y, test_size=0.2)
        assert not np.allclose(X_test.mean(axis=0), 0, atol=1e-6)

    def test_inverse_transform_roundtrip(self, features_and_target):
        X, y = features_and_target
        X_train, X_test, y_train, y_test, sX, sy = get_splits(X, y, test_size=0.2)
        recovered = sy.inverse_transform(y_test.reshape(-1, 1)).flatten()
        assert recovered.min() >= 400
        assert recovered.max() <= 36000


class TestModelOutput:
    def test_keras_model_exists(self):
        assert os.path.exists(os.path.join(MODELS_DIR, "best_nn.keras"))

    def test_keras_model_not_empty(self):
        p = os.path.join(MODELS_DIR, "best_nn.keras")
        if os.path.exists(p):
            assert os.path.getsize(p) > 10_000

    def test_scaler_y_pkl_exists(self):
        assert os.path.exists(os.path.join(MODELS_DIR, "scaler_y.pkl"))

    def test_tflite_exists(self):
        assert os.path.exists(os.path.join(MODELS_DIR, "model.tflite"))

    def test_tflite_not_empty(self):
        p = os.path.join(MODELS_DIR, "model.tflite")
        if os.path.exists(p):
            assert os.path.getsize(p) > 1000
