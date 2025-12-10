# src/evaluate.py
"""
Evaluation script for credit-risk-ml-system.

What it does:
- Loads a saved pipeline (models/pipeline.joblib) and threshold (models/threshold.json)
- Loads the raw CSV dataset, splits into train/test (stratified)
- Computes metrics on the test set: ROC-AUC, PR-AUC, Precision/Recall/F1 at saved threshold
- Plots and saves: ROC curve, Precision-Recall curve, Confusion Matrix
- Produces a SHAP summary plot (uses TreeExplainer for tree models) on a sample of test rows
- Saves metrics and plots to artifacts/

Usage:
    python src/evaluate.py --data-path data/raw/loan_default.csv

Notes:
- Make sure you ran `src/train.py` first and that models/pipeline.joblib and models/threshold.json exist.
- SHAP can be slow; we sample up to `--shap-sample` rows (default 1000).
"""

import argparse
import json
import os
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Try import shap; if not installed, skip SHAP section gracefully
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

from src.data_prep import prepare_features_and_target, get_feature_names


# ---------------------------
# Plot helpers
# ---------------------------

def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"PR curve (AP = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(cm: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    # annotate
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------
# Main evaluation
# ---------------------------

def evaluate_pipeline(
    data_path: str,
    pipeline_path: str = "models/pipeline.joblib",
    threshold_path: str = "models/threshold.json",
    out_dir: str = "artifacts",
    test_size: float = 0.2,
    random_state: int = 42,
    shap_sample: int = 1000,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1) load pipeline & threshold
    pipeline = joblib.load(pipeline_path)
    with open(threshold_path, "r") as fh:
        threshold = json.load(fh).get("threshold", 0.5)

    # 2) load raw data & prepare X,y
    df = pd.read_csv(data_path)
    X, y = prepare_features_and_target(df)

    # 3) train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 4) predict probabilities on test set (pipeline handles preprocessing)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # 5) metrics
    roc_auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)

    # 6) save metrics
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": float(threshold),
        "n_test": int(len(y_test)),
        "positive_rate_test": float((y_test == 1).mean()),
        "confusion_matrix": cm.tolist(),
    }

    metrics_path = os.path.join(out_dir, "evaluation_metrics.json")
    pd.Series(metrics).to_json(metrics_path)
    # also a human-readable CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(out_dir, "evaluation_metrics.csv"), index=False)

    # 7) plots
    plot_roc(y_test.values, y_proba, os.path.join(plots_dir, "roc_curve.png"))
    plot_pr(y_test.values, y_proba, os.path.join(plots_dir, "pr_curve.png"))
    plot_confusion(cm, os.path.join(plots_dir, "confusion_matrix.png"))

    # 8) SHAP explanations (best-effort; skip if shap not available)
    shap_info = {"shap_available": False, "shap_sample_used": 0}
    if _HAS_SHAP:
        try:
            # attempt to retrieve feature names and preprocessed data for SHAP
            # If the pipeline is Pipeline([("preprocessor", preproc), ("model", model)])
            # we will transform X_test with preprocessor to get numeric input for TreeExplainer.
            preprocessor = pipeline.named_steps.get("preprocessor", None)
            model = pipeline.named_steps.get("model", None)

            # get feature names via helper (works with our data_prep.get_feature_names)
            feature_names = []
            try:
                feature_names = get_feature_names(preprocessor)
            except Exception:
                # fallback: use X_test.columns (may be untransformed)
                feature_names = list(X_test.columns)

            # prepare background/sample for SHAP
            shap_sample_n = min(int(shap_sample), len(X_test))
            # sample without replacement for speed
            X_shap_sample = X_test.sample(n=shap_sample_n, random_state=random_state)

            # Transform to model input space if preprocessor exists
            if preprocessor is not None:
                X_shap_transformed = preprocessor.transform(X_shap_sample)
            else:
                # assume pipeline expects raw features
                X_shap_transformed = X_shap_sample.values

            # Use TreeExplainer for tree models (fast). Fallback to shap.Explainer if necessary.
            explainer = None
            try:
                # For xgboost/sklearn tree-based models
                if hasattr(model, "get_booster") or hasattr(model, "booster"):
                    # XGBoost compatible
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model, X_shap_transformed)
            except Exception:
                explainer = shap.Explainer(model, X_shap_transformed)

            shap_values = explainer.shap_values(X_shap_transformed)
            # shap_values shape depends on version/model; we try to plot summary
            plt.figure(figsize=(8, 6))
            try:
                # try summary_plot with matplotlib backend and save figure
                shap.summary_plot(shap_values, X_shap_transformed, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "shap_summary.png"), dpi=150)
                plt.close()
            except Exception:
                # fallback: convert to DataFrame and plot bar of mean abs shap
                importances = np.abs(shap_values).mean(axis=0)
                idx = np.argsort(importances)[::-1][:30]
                feat = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
                vals = importances[idx]
                plt.figure(figsize=(8, 6))
                plt.barh(range(len(vals))[::-1], vals)
                plt.yticks(range(len(vals))[::-1], feat)
                plt.title("Mean |SHAP value| (top features)")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "shap_summary_bar.png"), dpi=150)
                plt.close()

            shap_info["shap_available"] = True
            shap_info["shap_sample_used"] = shap_sample_n
        except Exception as e:
            shap_info["error"] = str(e)
    else:
        shap_info["note"] = "shap not installed"

    # 9) create a small report CSV of predictions (useful for debugging)
    report_df = X_test.copy().reset_index(drop=True)
    report_df["_y_true"] = y_test.reset_index(drop=True)
    report_df["_y_proba"] = y_proba
    report_df["_y_pred"] = y_pred
    report_df.to_csv(os.path.join(out_dir, "predictions_sample.csv"), index=False)

    # final results dict
    results = {"metrics": metrics, "plots_dir": plots_dir, "shap": shap_info}
    # save summary
    with open(os.path.join(out_dir, "evaluation_summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    return results


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved credit risk pipeline")
    parser.add_argument("--data-path", type=str, required=True, help="Path to raw CSV")
    parser.add_argument("--pipeline-path", type=str, default="models/pipeline.joblib", help="Saved pipeline path")
    parser.add_argument("--threshold-path", type=str, default="models/threshold.json", help="Saved threshold path")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Output directory for plots and metrics")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--shap-sample", type=int, default=1000, help="Max rows for SHAP sampling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_pipeline(
        data_path=args.data_path,
        pipeline_path=args.pipeline_path,
        threshold_path=args.threshold_path,
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        shap_sample=args.shap_sample,
    )
    print("Evaluation complete. Summary:")
    print(json.dumps(results["metrics"], indent=2))
    if results["shap"].get("shap_available"):
        print(f"SHAP summary saved. Sample used: {results['shap']['shap_sample_used']}")
    else:
        print("SHAP not available or failed — check artifacts for details.")
