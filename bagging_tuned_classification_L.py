#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version-adaptive Bagging classification with hyperparameter tuning.
- Auto target detection (or --target).
- OHE for categoricals, scaling for numerics.
- RandomizedSearchCV on f1_macro with Stratified CV.
- Prints results, saves all artifacts, and plots CM.

Run:
  python bagging_tuned_classification.py
Options:
  --data /path/to.csv
  --target TARGET_COL
  --outdir ./results_bagging
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
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt
import joblib


# ---------- Helpers ----------
def detect_target(df: pd.DataFrame, user_target: str | None) -> str:
    if user_target is not None:
        if user_target not in df.columns:
            raise ValueError(f"--target={user_target} not found in columns.")
        return user_target
    for c in ("target", "label", "class", "y", "output"):
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


def make_bagging(base_tree):
    """Return BaggingClassifier compatible with both old and new sklearn."""
    try:
        # Newer sklearn (>=1.2)
        return BaggingClassifier(estimator=base_tree, random_state=42, n_jobs=-1)
    except TypeError:
        # Older sklearn
        return BaggingClassifier(base_estimator=base_tree, random_state=42, n_jobs=-1)


def build_pipeline(cat_cols, num_cols):
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", make_ohe(), cat_cols),
            ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    base_tree = DecisionTreeClassifier(random_state=42)
    clf = make_bagging(base_tree)
    return Pipeline(steps=[("prep", preprocess), ("clf", clf)])


def raw_search_space():
    """Include both 'estimator' and 'base_estimator' keys; we'll filter by get_params."""
    return {
        # New API
        "clf__estimator__max_depth": [None, 5, 10, 15],
        "clf__estimator__min_samples_split": [2, 5, 10, 20],
        "clf__estimator__min_samples_leaf": [1, 2, 4, 8],
        # Old API
        "clf__base_estimator__max_depth": [None, 5, 10, 15],
        "clf__base_estimator__min_samples_split": [2, 5, 10, 20],
        "clf__base_estimator__min_samples_leaf": [1, 2, 4, 8],

        "clf__n_estimators": [50, 100, 150],
        "clf__max_samples": [0.7, 1.0],
        "clf__max_features": [0.7, 1.0],
        "clf__bootstrap": [True, False],
        "clf__bootstrap_features": [False, True],
    }


def filter_search_space_for(pipe: Pipeline, space: dict) -> dict:
    """Keep only keys that exist in pipe.get_params()."""
    valid = pipe.get_params(deep=True)
    return {k: v for k, v in space.items() if k in valid}


def plot_and_save_cm(cm: np.ndarray, labels: list[str], out_path: Path):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="build/classification_LEXPLORE.csv",
                        help="Path to CSV dataset.")
    parser.add_argument("--target", default=None, help="Target column name (optional).")
    parser.add_argument("--outdir", default="./results_classification_LEXPLORE", help="Output directory.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size (0-1).")
    parser.add_argument("--cv", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--n_iter", type=int, default=40, help="RandomizedSearch iterations.")
    args = parser.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data)
    target_col = detect_target(df, args.target)
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Split (stratify for classification)
    strat = y if type_of_target(y) in ("binary", "multiclass") else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=strat
    )

    # Column types
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Pipeline + search space
    pipe = build_pipeline(cat_cols, num_cols)
    space = filter_search_space_for(pipe, raw_search_space())

    # Hyperparameter search
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=space,
        n_iter=args.n_iter,
        scoring="f1_macro",
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

    # Probabilities (if available)
    y_proba = None
    try:
        y_proba = best_model.predict_proba(X_test)
    except Exception:
        pass

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    # ROC-AUC (binary or multiclass)
    rocauc = None
    try:
        ttype = type_of_target(y_test)
        if y_proba is not None:
            if ttype == "binary" and y_proba.shape[1] == 2:
                rocauc = roc_auc_score(y_test, y_proba[:, 1])
            elif ttype == "multiclass":
                rocauc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        pass

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = list(np.unique(y_test))

    # Save artifacts
    metrics = {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "roc_auc": rocauc,
        "best_cv_score_f1_macro": search.best_score_,
        "best_index_cv": int(search.best_index_),
        "target_column": target_col,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)

    with open(run_dir / "best_params.json", "w") as f:
        json.dump(search.best_params_, f, indent=2)

    report = classification_report(y_test, y_pred, digits=4)
    with open(run_dir / "classification_report.txt", "w") as f:
        f.write(report)

    pred_df = pd.DataFrame({"y_true": y_test.reset_index(drop=True), "y_pred": y_pred})
    if y_proba is not None:
        for i in range(y_proba.shape[1]):
            pred_df[f"proba_{i}"] = y_proba[:, i]
    pred_df.to_csv(run_dir / "predictions.csv", index=False)

    # Save model
    joblib.dump(best_model, run_dir / "bagging_model.joblib")

    # Plot CM
    plot_and_save_cm(cm, labels, run_dir / "confusion_matrix.png")

    # Console summary
    print("\n=== SUMMARY ===")
    print(f"Target column: {target_col}")
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print("\nBest Parameters (pipeline keys):")
    print(json.dumps(search.best_params_, indent=2))
    print("\nMetrics (Test):")
    print(pd.DataFrame([metrics]))
    print("\nClassification Report (Test):")
    print(report)
    print("\nSaved outputs in:", run_dir.resolve())
    for name in [
        "metrics.csv",
        "best_params.json",
        "classification_report.txt",
        "predictions.csv",
        "bagging_model.joblib",
        "confusion_matrix.png",
    ]:
        print(" -", name)


if __name__ == "__main__":
    main()
