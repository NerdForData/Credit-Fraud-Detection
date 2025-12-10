# src/train.py
"""
Train script for credit-risk-ml-system.

Creates a Pipeline(preprocessor + xgboost), runs Stratified K-Fold CV with threshold tuning,
persists the final pipeline and chosen threshold, and writes metrics to disk.

Example:
    python src/train.py --data-path data/raw/loan_default.csv
"""
import argparse
import json
import os
from typing import Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from src.data_prep import get_preprocessor, prepare_features_and_target, get_feature_names


# -------------------------
# Utility functions
# -------------------------

def compute_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict]:
    """
    Given true labels and predicted probabilities for positive class,
    compute PR curve and return the threshold that maximizes F1.
    Also return metrics at that threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns thresholds with length = len(precision)-1
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    if len(f1_scores) == 0:
        # fallback
        best_idx = 0
        best_threshold = 0.5
        best_metrics = {"precision": precision[0], "recall": recall[0], "f1": f1_scores[0] if len(f1_scores)>0 else 0.0}
    else:
        best_idx = int(np.nanargmax(f1_scores))
        best_threshold = thresholds[best_idx]
        best_metrics = {
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1": float(f1_scores[best_idx]),
        }
    return float(best_threshold), best_metrics


def evaluate_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict:
    """
    Compute binary metrics at the given threshold on probabilities for the positive class.
    """
    y_pred = (y_proba >= threshold).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "confusion_matrix": cm.tolist()}


# -------------------------
# Main training logic
# -------------------------

def run_training(
    data_path: str,
    model_out_path: str = "models/pipeline.joblib",
    threshold_out_path: str = "models/threshold.json",
    metrics_out_path: str = "artifacts/metrics.csv",
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
):
    os.makedirs(os.path.dirname(model_out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(threshold_out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metrics_out_path) or "artifacts", exist_ok=True)

    # 1) Load raw data
    df = pd.read_csv(data_path)
    X, y = prepare_features_and_target(df)

    # Quick sanity
    assert len(X) == len(y), "Features and target lengths must match."
    print(f"Loaded data: {len(X)} rows, {X.shape[1]} features")

    # 2) Build pipeline: preprocessor + XGB
    preprocessor = get_preprocessor()
    # compute scale_pos_weight for imbalanced dataset
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    scale_pos_weight = neg / (pos + 1e-9)
    print(f"Positive / Negative counts: {pos} / {neg}  -> scale_pos_weight={scale_pos_weight:.2f}")

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        verbosity=0,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb),
    ])

    # 3) CV: Stratified K-Fold with threshold tuning per fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results: List[Dict] = []
    best_thresholds: List[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"Training fold {fold_idx}/{n_splits} ...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)

        # get validation probabilities for positive class
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]

        # global metrics
        roc = float(roc_auc_score(y_val, y_val_proba))
        pr_auc = float(average_precision_score(y_val, y_val_proba))

        # threshold tuning on PR curve (choose threshold that maximizes F1)
        best_thr, best_metrics = compute_best_threshold(y_val.values, y_val_proba)
        eval_metrics = evaluate_at_threshold(y_val.values, y_val_proba, best_thr)

        fold_result = {
            "fold": fold_idx,
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "best_threshold": best_thr,
            "best_precision_on_pr": best_metrics.get("precision"),
            "best_recall_on_pr": best_metrics.get("recall"),
            "best_f1_on_pr": best_metrics.get("f1"),
            "precision_at_best_threshold": eval_metrics["precision"],
            "recall_at_best_threshold": eval_metrics["recall"],
            "f1_at_best_threshold": eval_metrics["f1"],
            "confusion_matrix": eval_metrics["confusion_matrix"],
        }
        fold_results.append(fold_result)
        best_thresholds.append(best_thr)

        print(
            f" Fold {fold_idx} - ROC-AUC: {roc:.4f}  PR-AUC: {pr_auc:.4f}  "
            f"best_thr: {best_thr:.4f}  f1: {eval_metrics['f1']:.4f}"
        )

    # 4) Aggregate results and save metrics
    metrics_df = pd.DataFrame(fold_results)
    metrics_df.to_csv(metrics_out_path, index=False)
    print(f"Saved CV fold metrics to {metrics_out_path}")

    # 5) Final training on full dataset
    print("Training final pipeline on full dataset ...")
    pipeline.fit(X, y)

    # Choose final threshold as median of best fold thresholds (robust)
    if len(best_thresholds) > 0:
        final_threshold = float(np.median(best_thresholds))
    else:
        final_threshold = 0.5

    # Save the pipeline and threshold
    joblib.dump(pipeline, model_out_path)
    with open(threshold_out_path, "w") as fh:
        json.dump({"threshold": final_threshold}, fh)

    print(f"Saved final pipeline to {model_out_path}")
    print(f"Saved chosen threshold to {threshold_out_path} (threshold={final_threshold:.4f})")

    # 6) Print aggregated CV summary
    print("\nCV summary (mean ± std):")
    print(f"ROC-AUC: {metrics_df['roc_auc'].mean():.4f} ± {metrics_df['roc_auc'].std():.4f}")
    print(f"PR-AUC:  {metrics_df['pr_auc'].mean():.4f} ± {metrics_df['pr_auc'].std():.4f}")
    print(f"F1@thr:  {metrics_df['f1_at_best_threshold'].mean():.4f} ± {metrics_df['f1_at_best_threshold'].std():.4f}")

    return {
        "metrics_df": metrics_df,
        "pipeline_path": model_out_path,
        "threshold": final_threshold,
    }


# -------------------------
# CLI
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train credit-risk-ml-system pipeline")
    parser.add_argument("--data-path", type=str, required=True, help="Path to raw CSV file")
    parser.add_argument("--model-out", type=str, default="models/pipeline.joblib", help="Output path for pipeline")
    parser.add_argument("--threshold-out", type=str, default="models/threshold.json", help="Output path for chosen threshold")
    parser.add_argument("--metrics-out", type=str, default="artifacts/metrics.csv", help="Output path for CV metrics")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n-estimators", type=int, default=200, help="XGBoost n_estimators")
    parser.add_argument("--max-depth", type=int, default=6, help="XGBoost max_depth")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="XGBoost learning_rate")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        data_path=args.data_path,
        model_out_path=args.model_out,
        threshold_out_path=args.threshold_out,
        metrics_out_path=args.metrics_out,
        n_splits=args.n_splits,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )
