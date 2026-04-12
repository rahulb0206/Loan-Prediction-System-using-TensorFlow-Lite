"""
predict.py
----------
Run a single loan amount prediction from the command line.

Example:
    python src/predict.py \
        --age 28 \
        --income 55000 \
        --emp_length 3 \
        --int_rate 11.5 \
        --cred_hist 4 \
        --home_ownership RENT \
        --loan_intent PERSONAL \
        --loan_grade B \
        --default_on_file N
"""

import argparse
import os
import pickle
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "models")

HOME_OPTS   = ["RENT", "MORTGAGE", "OWN", "OTHER"]
INTENT_OPTS = ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"]
GRADE_OPTS  = ["A", "B", "C", "D", "E", "F", "G"]


def build_input_vector(age, income, emp_length, int_rate, cred_hist,
                       home_ownership, loan_intent, loan_grade, default_on_file):
    """Build the 20-feature one-hot encoded input row matching training feature order."""
    row = [
        age,
        income,
        emp_length,
        int_rate,
        cred_hist,
        1.0 if home_ownership == "OTHER"    else 0.0,
        1.0 if home_ownership == "OWN"      else 0.0,
        1.0 if home_ownership == "RENT"     else 0.0,
        1.0 if loan_intent == "EDUCATION"       else 0.0,
        1.0 if loan_intent == "HOMEIMPROVEMENT" else 0.0,
        1.0 if loan_intent == "MEDICAL"         else 0.0,
        1.0 if loan_intent == "PERSONAL"        else 0.0,
        1.0 if loan_intent == "VENTURE"         else 0.0,
        1.0 if loan_grade == "B" else 0.0,
        1.0 if loan_grade == "C" else 0.0,
        1.0 if loan_grade == "D" else 0.0,
        1.0 if loan_grade == "E" else 0.0,
        1.0 if loan_grade == "F" else 0.0,
        1.0 if loan_grade == "G" else 0.0,
        1.0 if default_on_file == "Y" else 0.0,
    ]
    return np.array(row).reshape(1, -1)


def predict(age, income, emp_length, int_rate, cred_hist,
            home_ownership, loan_intent, loan_grade, default_on_file,
            model_type="nn"):

    with open(os.path.join(MODELS_DIR, "scaler_X.pkl"), "rb") as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler_y.pkl"), "rb") as f:
        scaler_y = pickle.load(f)

    import pandas as pd
    feature_names = scaler_X.feature_names_in_
    X_raw = build_input_vector(age, income, emp_length, int_rate, cred_hist,
                               home_ownership, loan_intent, loan_grade, default_on_file)
    X_df = pd.DataFrame(X_raw, columns=feature_names)
    X_scaled = scaler_X.transform(X_df)

    if model_type == "nn":
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "best_nn.keras"))
        pred_scaled = model.predict(X_scaled, verbose=0).flatten()[0]
        pred_dollars = scaler_y.inverse_transform([[pred_scaled]])[0][0]

    elif model_type == "lr":
        with open(os.path.join(MODELS_DIR, "linear_regression.pkl"), "rb") as f:
            lr = pickle.load(f)
        pred_dollars = lr.predict(X_scaled)[0]

    elif model_type == "dummy":
        with open(os.path.join(MODELS_DIR, "dummy.pkl"), "rb") as f:
            dummy = pickle.load(f)
        pred_scaled = dummy.predict(X_scaled)[0]
        pred_dollars = scaler_y.inverse_transform([[pred_scaled]])[0][0]

    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose: nn, lr, dummy")

    return pred_dollars


def main():
    parser = argparse.ArgumentParser(description="Predict loan amount for a single borrower profile")
    parser.add_argument("--age",           type=float, required=True,  help="Age (20–65)")
    parser.add_argument("--income",        type=float, required=True,  help="Annual income in USD")
    parser.add_argument("--emp_length",    type=float, required=True,  help="Employment length in years (0–41)")
    parser.add_argument("--int_rate",      type=float, required=True,  help="Loan interest rate %% (5.4–23.2)")
    parser.add_argument("--cred_hist",     type=float, required=True,  help="Credit history length in years (2–30)")
    parser.add_argument("--home_ownership",type=str,   required=True,  choices=HOME_OPTS)
    parser.add_argument("--loan_intent",   type=str,   required=True,  choices=INTENT_OPTS)
    parser.add_argument("--loan_grade",    type=str,   required=True,  choices=GRADE_OPTS)
    parser.add_argument("--default_on_file", type=str, required=True,  choices=["Y", "N"])
    parser.add_argument("--model",         type=str,   default="nn",   choices=["nn", "lr", "dummy"],
                        help="Which model to use (default: nn)")
    args = parser.parse_args()

    print("\nInput profile:")
    print(f"  Age: {args.age}  |  Income: ${args.income:,.0f}  |  Employment: {args.emp_length} yrs")
    print(f"  Interest rate: {args.int_rate}%  |  Credit history: {args.cred_hist} yrs")
    print(f"  Home: {args.home_ownership}  |  Intent: {args.loan_intent}  |  Grade: {args.loan_grade}  |  Prior default: {args.default_on_file}")
    print(f"  Model: {args.model}")

    result = predict(
        age=args.age,
        income=args.income,
        emp_length=args.emp_length,
        int_rate=args.int_rate,
        cred_hist=args.cred_hist,
        home_ownership=args.home_ownership,
        loan_intent=args.loan_intent,
        loan_grade=args.loan_grade,
        default_on_file=args.default_on_file,
        model_type=args.model,
    )

    print(f"\n  Predicted loan amount: ${result:,.0f}\n")


if __name__ == "__main__":
    main()
