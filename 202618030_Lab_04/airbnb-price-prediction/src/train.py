"""
Training script -- trains Linear Regression and XGBoost regression pipelines
on the cleaned Airbnb dataset, compares them, and saves the best pipeline
(preprocessing + model bundled together) to models/best_model.joblib.

This exact logic is also walked through step-by-step, with explanations,
in notebooks/01_airbnb_price_prediction.ipynb. Run this script directly if
you just want to (re)generate the model artifact quickly:

    python src/train.py
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    clean_raw_data,
    get_feature_target,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "AB_NYC_2019.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
    }


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    clean = clean_raw_data(df)
    X, y = get_feature_target(clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    results = {}

    # ---------------- Linear Regression ----------------
    lr_pipeline = Pipeline(
        steps=[("preprocess", build_preprocessor()), ("model", LinearRegression())]
    )
    lr_pipeline.fit(X_train, y_train)
    results["LinearRegression"] = evaluate(lr_pipeline, X_train, y_train, X_test, y_test)
    print("Linear Regression:", results["LinearRegression"])

    best_name = "LinearRegression"
    best_pipeline = lr_pipeline

    # ---------------- XGBoost ----------------
    try:
        from xgboost import XGBRegressor

        xgb_pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    XGBRegressor(
                        random_state=42,
                        n_estimators=400,
                        objective="reg:squarederror",
                    ),
                ),
            ]
        )

        param_grid = {
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.1],
            "model__subsample": [0.8, 1.0],
        }
        search = GridSearchCV(
            xgb_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1
        )
        search.fit(X_train, y_train)
        xgb_best = search.best_estimator_
        print("Best XGBoost params:", search.best_params_)

        results["XGBoost"] = evaluate(xgb_best, X_train, y_train, X_test, y_test)
        print("XGBoost:", results["XGBoost"])

        if results["XGBoost"]["test_rmse"] < results["LinearRegression"]["test_rmse"]:
            best_name = "XGBoost"
            best_pipeline = xgb_best

    except ImportError:
        print(
            "\n[WARNING] xgboost is not installed in this environment, so only "
            "Linear Regression was trained. Install xgboost (see requirements.txt) "
            "and re-run this script to train/compare XGBoost as well.\n"
        )

    print(f"\nBest model: {best_name}")
    joblib.dump(best_pipeline, MODEL_DIR / "best_model.joblib")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"Saved pipeline to {MODEL_DIR / 'best_model.joblib'}")
    print(f"Saved metrics to {MODEL_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
