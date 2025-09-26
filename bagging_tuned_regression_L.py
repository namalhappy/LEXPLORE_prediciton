#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version-adaptive Bagging regression with hyperparameter tuning.
- Auto-detects target (or use --target)
- OHE for categoricals, scaling for numerics
- RandomizedSearchCV (RMSE-focused) with Stratified-like robustness not required (regression)
- Prints results, saves artifacts, and plots parity & residuals

Run:
  python bagging_tuned_regression.py
Options:
  --data /path/to.csv
  --target TARGET_COL
  --outdir ./results_bagging_reg
  --test_size 0.2
  --cv 5
  --n_iter 40
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error
)
import matplotlib.pyplot as plt
import joblib


# ---------- Helpers ----------
def detect_target(df: pd.DataFrame, user_target: str | None) -> str:
    if user_target is not None:
        if user_target not in df.columns:
            raise ValueError(f"--target={user_target} not found in columns.")
        return user_target
    for c in ("target", "label", "y", "output"):
        if c in df.columns:
            return c
    return df.columns[-1]


def make_ohe():
    """Return OneHotEncoder compatible with both old and new sklearn."""
    try:
        # Newer sklearn (>=1.2)
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_bagging_reg(base_tree):
    """Return BaggingRegressor compatible with both old and new sklearn."""
    try:
        # Newer sklearn (>=1.2)
        return BaggingRegressor(estimator=base_tree, random_state=42, n_jobs=-1)
    except TypeError:
        # Older sklearn
        return BaggingRegressor(base_estimator=base_tree, random_state=42, n_jobs=-1)


def build_pipeline(cat_cols, num_cols):
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", make_ohe(), cat_cols),
            ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    base_tree = DecisionTreeRegressor(random_state=42)
    reg = make_bagging_reg(base_tree)
    return Pipeline(steps=[("prep", preprocess), ("reg", reg)])


def raw_search_space():
    """Include both new (estimator__) and old (base_estimator__) keys; we'll filter by get_params()."""
    return {
        # New API keys
        "reg__estimator__max_depth": [None, 5, 10, 15],
        "reg__estimator__min_samples_split": [2, 5, 10, 20],
        "reg__estimator__min_samples_leaf": [1, 2, 4, 8],
        # Old API keys
        "reg__base_estimator__max_depth": [None, 5, 10, 15],
        "reg__base_estimator__min_samples_split": [2, 5, 10, 20],
        "reg__base_estimator__min_samples_leaf": [1, 2, 4, 8],

        "reg__n_estimators": [50, 100, 150],
        "reg__max_samples": [0.7, 1.0],
        "reg__max_features": [0.7, 1.0],
        "reg__bootstrap": [True, False],
        "reg__bootstrap_features": [False, True],
    }


def filter_search_space_for(pipe: Pipeline, space: dict) -> dict:
    params = pipe.get_params(deep=True)
    return {k: v for k, v in space.items() if k in params}


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.finfo(float).eps, np.abs(y_true)))) * 100.0
    return dict(MAE=mae, MSE=mse, RMSE=rmse, R2=r2, MAPE=mape)


def plot_and_save_parity(y_true, y_pred, out_path: Path, title="Prediction Parity (y vs ŷ)"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors="none")
    # y=x line
    lims = [
        np.min([plt.gca().get_xlim(), plt.gca().get_ylim()]),
        np.max([plt.gca().get_xlim(), plt.gca().get_ylim()])]
    plt.plot(lims, lims)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_and_save_residuals(y_true, y_pred, out_path: Path, title="Residuals vs Predicted"):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors="none")
    plt.axhline(0.0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (y - ŷ)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="build/regression_LEXPLORE.csv",
                        help="Path to CSV dataset.")
    parser.add_argument("--target", default=None, help="Target column name (optional).")
    parser.add_argument("--outdir", default="./results_regression_LEXPLORE", help="Output directory.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size (0-1).")
    parser.add_argument("--cv", type=int, default=5, help="KFold CV splits.")
    parser.add_argument("--n_iter", type=int, default=40, help="RandomizedSearch iterations.")
    args = parser.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data)
    target_col = detect_target(df, args.target)
    y = df[target_col].astype(float)  # ensure numeric target
    X = df.drop(columns=[target_col])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    # Column types
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Pipeline + tuned search
    pipe = build_pipeline(cat_cols, num_cols)
    space = filter_search_space_for(pipe, raw_search_space())

    # RMSE-focused (sklearn uses "neg_root_mean_squared_error" — higher is better)
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=space,
        n_iter=args.n_iter,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        cv=cv,
        random_state=42,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Predict
    y_pred = best_model.predict(X_test)

    # Metrics
    mets = regression_metrics(y_test, y_pred)
    mets["best_cv_score_negRMSE"] = float(search.best_score_)  # higher is better
    mets["target_column"] = target_col
    mets["n_train"] = int(len(y_train))
    mets["n_test"] = int(len(y_test))

    # Save artifacts
    pd.DataFrame([mets]).to_csv(run_dir / "metrics.csv", index=False)

    with open(run_dir / "best_params.json", "w") as f:
        json.dump(search.best_params_, f, indent=2)

    pred_df = pd.DataFrame({
        "y_true": y_test.reset_index(drop=True),
        "y_pred": y_pred
    })
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    # Save model
    joblib.dump(best_model, run_dir / "bagging_regressor.joblib")

    # Plots
    plot_and_save_parity(y_test.values, y_pred, run_dir / "parity_plot.png")
    plot_and_save_residuals(y_test.values, y_pred, run_dir / "residuals_vs_pred.png")

    # Console summary
    print("\n=== SUMMARY (Regression) ===")
    print(f"Target column: {target_col}")
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print("\nBest Parameters (pipeline keys):")
    print(json.dumps(search.best_params_, indent=2))
    print("\nMetrics (Test):")
    print(pd.DataFrame([mets]))
    print("\nSaved outputs in:", run_dir.resolve())
    for name in [
        "metrics.csv",
        "best_params.json",
        "predictions.csv",
        "bagging_regressor.joblib",
        "parity_plot.png",
        "residuals_vs_pred.png",
    ]:
        print(" -", name)


if __name__ == "__main__":
    main()
