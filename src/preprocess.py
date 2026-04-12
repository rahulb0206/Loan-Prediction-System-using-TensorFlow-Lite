"""
preprocess.py
-------------
Load, clean, and prepare the credit risk dataset for regression.
Target: loan_amnt (continuous, $500–$35,000)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df["person_emp_length"] = df["person_emp_length"].fillna(
        df["person_emp_length"].median()
    )

    # Interest rate missing values are grade-dependent, so group median makes more sense
    df["loan_int_rate"] = df["loan_int_rate"].fillna(
        df.groupby("loan_grade")["loan_int_rate"].transform("median")
    )

    assert df.isnull().sum().sum() == 0
    return df


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cat_cols = [
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # loan_percent_income = loan_amnt / person_income — keeping it would leak the target
    # loan_status is only known after repayment, not at application time
    X = df_enc.drop(columns=["loan_amnt", "loan_percent_income", "loan_status"])
    y = df_enc["loan_amnt"]

    return X, y


def get_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    save_scalers_path: str = None,
) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler_X = StandardScaler()
    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc = scaler_X.transform(X_test)

    # Scale y too — loan amounts range $500–$35k, unscaled MSE runs into the billions
    scaler_y = StandardScaler()
    y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_sc = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    if save_scalers_path:
        os.makedirs(save_scalers_path, exist_ok=True)
        with open(os.path.join(save_scalers_path, "scaler_X.pkl"), "wb") as f:
            pickle.dump(scaler_X, f)
        with open(os.path.join(save_scalers_path, "scaler_y.pkl"), "wb") as f:
            pickle.dump(scaler_y, f)

    return X_train_sc, X_test_sc, y_train_sc, y_test_sc, scaler_X, scaler_y
