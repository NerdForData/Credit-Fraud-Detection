# src/evaluate.py
"""
Model Evaluation Pipeline for Credit Risk Prediction System.

This script evaluates a trained model's performance on held-out test data,
generates comprehensive visualizations, and creates SHAP explanations to
understand feature importance and model decisions.

Core Functionality:
    - Load pre-trained pipeline and optimized threshold
    - Split data into train/test sets (stratified)
    - Compute comprehensive metrics on test set
    - Generate diagnostic plots (ROC, PR, confusion matrix)
    - Create SHAP explanations for model interpretability
    - Save evaluation artifacts for reporting

Key Metrics Computed:
    - ROC-AUC: Overall ability to distinguish classes
    - PR-AUC: Performance focus on positive class (imbalanced data)
    - Precision: How many predicted defaults are correct?
    - Recall: How many actual defaults did we catch?
    - F1 Score: Harmonic mean of precision and recall
    - Confusion Matrix: Breakdown of TP, TN, FP, FN

Visualization Outputs:
    - ROC Curve: Trade-off between TPR and FPR
    - Precision-Recall Curve: Better for imbalanced datasets
    - Confusion Matrix: Actual vs predicted class distribution
    - SHAP Summary Plot: Feature importance and impact direction

Usage:
    # Basic evaluation with default settings
    python -m src.evaluate --data-path data/raw/Loan_default.csv
    
    # Custom test split and SHAP sample size
    python -m src.evaluate --data-path data/raw/Loan_default.csv \
        --test-size 0.3 --shap-sample 2000
    
Prerequisites:
    - Must run src/train.py first to generate:
      * models/pipeline.joblib (trained model)
      * models/threshold.json (optimized threshold)
    - SHAP package optional but recommended for explainability

Output Files:
    - artifacts/evaluation_metrics.json: All metrics in JSON format
    - artifacts/evaluation_metrics.csv: Metrics in CSV format
    - artifacts/evaluation_summary.json: Complete evaluation summary
    - artifacts/predictions_sample.csv: Test set with predictions
    - artifacts/plots/roc_curve.png: ROC curve visualization
    - artifacts/plots/pr_curve.png: Precision-Recall curve
    - artifacts/plots/confusion_matrix.png: Confusion matrix heatmap
    - artifacts/plots/shap_summary.png: SHAP feature importance

Example:
    >>> from src.evaluate import evaluate_pipeline
    >>> results = evaluate_pipeline('data/raw/Loan_default.csv')
    >>> print(f"Test ROC-AUC: {results['metrics']['roc_auc']:.4f}")
    >>> print(f"Test F1: {results['metrics']['f1']:.4f}")
    
Notes:
    - Test set is stratified to maintain class distribution
    - SHAP computation can be slow; adjust --shap-sample for speed
    - SHAP uses TreeExplainer for XGBoost (fast and accurate)
    - All plots saved at 150 DPI for publication quality
"""

# Standard library imports
import argparse
import json
import os
from typing import Tuple

# Third-party imports
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Scikit-learn imports
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

# Try import SHAP; gracefully handle if not installed
# SHAP provides model interpretability through Shapley values
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

# Local imports
from src.data_prep import prepare_features_and_target, get_feature_names


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
# These functions generate publication-quality diagnostic plots to visualize
# model performance from different perspectives.

def plot_roc(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    """
    Generate and save ROC (Receiver Operating Characteristic) curve.
    
    The ROC curve plots True Positive Rate (TPR) vs False Positive Rate (FPR)
    across all possible thresholds. It shows the trade-off between sensitivity
    and specificity. The diagonal line represents random guessing.
    
    Interpretation:
        - AUC = 1.0: Perfect classifier
        - AUC = 0.9-1.0: Excellent
        - AUC = 0.8-0.9: Good
        - AUC = 0.7-0.8: Fair
        - AUC = 0.5: Random guessing
    
    Args:
        y_true (np.ndarray): True binary labels (0 or 1).
            Shape: (n_samples,)
        y_proba (np.ndarray): Predicted probabilities for positive class.
            Shape: (n_samples,)
        out_path (str): File path to save the plot (e.g., 'roc_curve.png').
    
    Side Effects:
        - Creates PNG file at out_path
        - Closes matplotlib figure after saving
    
    Example:
        >>> plot_roc(y_test, y_pred_proba, 'artifacts/plots/roc.png')
    """
    # Compute ROC curve points
    # fpr: False positive rates at each threshold
    # tpr: True positive rates at each threshold
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    # Calculate area under ROC curve
    auc = roc_auc_score(y_true, y_proba)
    
    # Create figure
    plt.figure(figsize=(6, 6))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})", linewidth=2)
    
    # Plot diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    
    # Formatting
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save and close
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    """
    Generate and save Precision-Recall curve.
    
    The PR curve is more informative than ROC for imbalanced datasets because
    it focuses on the positive class. It shows the trade-off between precision
    (how many predicted positives are correct) and recall (how many actual
    positives we caught).
    
    Why PR curve for imbalanced data?
        - ROC can be overly optimistic when negatives dominate
        - PR curve highlights performance on the minority class
        - Average Precision (AP) is the area under PR curve
    
    Args:
        y_true (np.ndarray): True binary labels (0 or 1).
            Shape: (n_samples,)
        y_proba (np.ndarray): Predicted probabilities for positive class.
            Shape: (n_samples,)
        out_path (str): File path to save the plot.
    
    Side Effects:
        - Creates PNG file at out_path
        - Closes matplotlib figure after saving
    
    Example:
        >>> plot_pr(y_test, y_pred_proba, 'artifacts/plots/pr_curve.png')
        
    Notes:
        - Higher AP (Average Precision) is better
        - Baseline is the positive class ratio (e.g., 0.1 if 10% positive)
    """
    # Compute precision-recall curve
    # precision[i], recall[i] correspond to threshold thresholds[i]
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    # Calculate average precision (area under PR curve)
    # AP is a weighted mean of precisions at each threshold
    ap = average_precision_score(y_true, y_proba)
    
    # Create figure
    plt.figure(figsize=(6, 6))
    
    # Plot PR curve
    plt.plot(recall, precision, label=f"PR curve (AP = {ap:.4f})", linewidth=2)
    
    # Formatting
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save and close
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(cm: np.ndarray, out_path: str) -> None:
    """
    Generate and save confusion matrix heatmap.
    
    The confusion matrix shows the breakdown of predictions:
        [[TN, FP],
         [FN, TP]]
    
    Where:
        - TN (True Negative): Correctly predicted no default
        - FP (False Positive): Predicted default, but didn't default (Type I error)
        - FN (False Negative): Predicted no default, but did default (Type II error)
        - TP (True Positive): Correctly predicted default
    
    Business Impact:
        - FP: Rejected loan applicant who would have paid (lost revenue)
        - FN: Approved loan applicant who defaults (direct loss)
        - Which error is more costly depends on business context
    
    Args:
        cm (np.ndarray): Confusion matrix from sklearn.metrics.confusion_matrix.
            Shape: (2, 2) for binary classification
            Format: [[TN, FP], [FN, TP]]
        out_path (str): File path to save the plot.
    
    Side Effects:
        - Creates PNG file at out_path
        - Closes matplotlib figure after saving
    
    Example:
        >>> cm = confusion_matrix(y_test, y_pred)
        >>> plot_confusion(cm, 'artifacts/plots/cm.png')
    """
    # Create figure with square aspect ratio
    plt.figure(figsize=(4, 4))
    
    # Display confusion matrix as heatmap
    plt.imshow(cm, cmap=plt.cm.Blues, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    # Set tick labels
    ticks = np.arange(cm.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    
    # Annotate each cell with count
    # Use white text for dark cells, black for light cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight='bold'
            )
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



# ============================================================================
# MAIN EVALUATION PIPELINE
# ============================================================================
# This section orchestrates the complete evaluation workflow including
# metric computation, visualization, and SHAP explainability.

def evaluate_pipeline(
    data_path: str,
    pipeline_path: str = "models/pipeline.joblib",
    threshold_path: str = "models/threshold.json",
    out_dir: str = "artifacts",
    test_size: float = 0.2,
    random_state: int = 42,
    shap_sample: int = 1000,
) -> dict:
    """
    Comprehensive model evaluation with metrics, plots, and SHAP explanations.
    
    This function orchestrates the complete evaluation workflow:
    1. Load trained pipeline and optimized threshold
    2. Prepare data and create stratified train/test split
    3. Generate predictions on test set
    4. Compute comprehensive metrics
    5. Create diagnostic plots (ROC, PR, confusion matrix)
    6. Generate SHAP explanations for interpretability
    7. Save all artifacts to disk
    
    Evaluation Strategy:
        - Stratified split ensures balanced class distribution in test set
        - Test set is completely held out (not used during training)
        - Metrics computed at the threshold optimized during training
        - SHAP sampling for computational efficiency on large datasets
    
    Args:
        data_path (str): Path to raw CSV file with loan data.
            Should be the same data used for training
        pipeline_path (str): Path to trained pipeline.joblib file.
            Default: "models/pipeline.joblib"
        threshold_path (str): Path to threshold.json file.
            Default: "models/threshold.json"
        out_dir (str): Directory to save evaluation artifacts.
            Default: "artifacts"
        test_size (float): Fraction of data to use for testing.
            Default: 0.2 (20% test, 80% train)
        random_state (int): Random seed for reproducible splits.
            Default: 42
        shap_sample (int): Maximum number of samples for SHAP computation.
            Default: 1000 (balance between accuracy and speed)
    
    Returns:
        dict: Evaluation results with keys:
            - 'metrics' (Dict): All computed metrics
                * 'roc_auc': ROC-AUC score
                * 'pr_auc': Precision-Recall AUC (Average Precision)
                * 'precision': Precision at threshold
                * 'recall': Recall at threshold
                * 'f1': F1 score at threshold
                * 'threshold': Classification threshold used
                * 'n_test': Number of test samples
                * 'positive_rate_test': Positive class rate in test set
                * 'confusion_matrix': [[TN, FP], [FN, TP]]
            - 'plots_dir' (str): Directory containing saved plots
            - 'shap' (Dict): SHAP computation status and info
    
    Side Effects:
        - Creates output directories if they don't exist
        - Saves multiple files:
          * evaluation_metrics.json: Metrics in JSON format
          * evaluation_metrics.csv: Metrics in CSV format
          * evaluation_summary.json: Complete evaluation summary
          * predictions_sample.csv: Test predictions with features
          * plots/roc_curve.png: ROC curve visualization
          * plots/pr_curve.png: Precision-Recall curve
          * plots/confusion_matrix.png: Confusion matrix heatmap
          * plots/shap_summary.png: SHAP feature importance (if available)
    
    Example:
        >>> results = evaluate_pipeline(
        ...     data_path='data/raw/Loan_default.csv',
        ...     test_size=0.3,
        ...     shap_sample=500
        ... )
        >>> print(f"Test ROC-AUC: {results['metrics']['roc_auc']:.4f}")
        >>> print(f"Test F1: {results['metrics']['f1']:.4f}")
        
    Notes:
        - Requires models/pipeline.joblib and models/threshold.json to exist
        - SHAP computation is optional; evaluation continues if SHAP fails
        - Uses TreeExplainer for XGBoost (much faster than KernelExplainer)
        - All plots saved at 150 DPI for publication quality
    
    See Also:
        plot_roc: ROC curve plotting function
        plot_pr: Precision-Recall curve plotting function
        plot_confusion: Confusion matrix plotting function
    """
    # ========================================================================
    # STEP 1: SETUP OUTPUT DIRECTORIES
    # ========================================================================
    # Create artifacts directory structure
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # ========================================================================
    # STEP 2: LOAD TRAINED MODEL AND THRESHOLD
    # ========================================================================
    print(f"Loading pipeline from: {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    
    print(f"Loading threshold from: {threshold_path}")
    with open(threshold_path, "r") as fh:
        threshold = json.load(fh).get("threshold", 0.5)
    print(f"Using classification threshold: {threshold:.4f}")

    # ========================================================================
    # STEP 3: LOAD AND PREPARE DATA
    # ========================================================================
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare features and target using same preprocessing as training
    X, y = prepare_features_and_target(df)
    print(f"Prepared data: {len(X)} total samples")

    # ========================================================================
    # STEP 4: TRAIN/TEST SPLIT (STRATIFIED)
    # ========================================================================
    # Why stratified?
    # - Maintains same positive/negative ratio in train and test sets
    # - Critical for imbalanced datasets to get representative test metrics
    # - Ensures test set is a good proxy for production data
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,      # Fraction for test set
        stratify=y,                # Maintain class distribution
        random_state=random_state  # Reproducibility
    )
    
    print(f"Train set: {len(X_train)} samples ({(y_train==1).sum()} positives)")
    print(f"Test set:  {len(X_test)} samples ({(y_test==1).sum()} positives)")

    # ========================================================================
    # STEP 5: GENERATE PREDICTIONS
    # ========================================================================
    # Pipeline automatically applies preprocessing before prediction
    # predict_proba returns shape (n_samples, 2): [prob_class_0, prob_class_1]
    # We take [:, 1] to get only positive class probabilities
    
    print("\nGenerating predictions on test set...")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Convert probabilities to binary predictions using optimized threshold
    # If probability >= threshold, predict 1 (default), else 0 (no default)
    y_pred = (y_proba >= threshold).astype(int)
    
    print(f"Predicted {y_pred.sum()} positives out of {len(y_pred)} test samples")

    # ========================================================================
    # STEP 6: COMPUTE METRICS
    # ========================================================================
    print("\nComputing evaluation metrics...")
    
    # Threshold-independent metrics (use probabilities)
    roc_auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))
    
    # Threshold-dependent metrics (use binary predictions)
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # ========================================================================
    # STEP 7: ORGANIZE METRICS
    # ========================================================================
    metrics = {
        # Performance metrics
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        
        # Configuration
        "threshold": float(threshold),
        
        # Dataset statistics
        "n_test": int(len(y_test)),
        "positive_rate_test": float((y_test == 1).mean()),
        
        # Confusion matrix [[TN, FP], [FN, TP]]
        "confusion_matrix": cm.tolist(),
    }
    
    print(f"\nTest Set Metrics:")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")

    # ========================================================================
    # STEP 8: SAVE METRICS
    # ========================================================================
    # Save as JSON for programmatic access
    metrics_path = os.path.join(out_dir, "evaluation_metrics.json")
    pd.Series(metrics).to_json(metrics_path)
    
    # Also save as CSV for easy viewing
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(out_dir, "evaluation_metrics.csv"), index=False)
    
    print(f"\n✓ Saved metrics to {out_dir}/")

    # ========================================================================
    # STEP 9: GENERATE DIAGNOSTIC PLOTS
    # ========================================================================
    print(f"\nGenerating diagnostic plots...")
    
    # ROC Curve: Shows TPR vs FPR trade-off
    plot_roc(y_test.values, y_proba, os.path.join(plots_dir, "roc_curve.png"))
    
    # Precision-Recall Curve: Better for imbalanced data
    plot_pr(y_test.values, y_proba, os.path.join(plots_dir, "pr_curve.png"))
    
    # Confusion Matrix: Shows error breakdown
    plot_confusion(cm, os.path.join(plots_dir, "confusion_matrix.png"))
    
    print(f"✓ Saved plots to {plots_dir}/")

    # ========================================================================
    # STEP 10: SHAP EXPLANATIONS (OPTIONAL)
    # ========================================================================
    # SHAP (SHapley Additive exPlanations) provides feature importance
    # and shows how each feature contributes to individual predictions
    
    shap_info = {"shap_available": False, "shap_sample_used": 0}
    
    if _HAS_SHAP:
        print(f"\nComputing SHAP explanations...")
        try:
            # Extract pipeline components
            # Pipeline structure: [("preprocessor", preproc), ("model", xgb)]
            preprocessor = pipeline.named_steps.get("preprocessor", None)
            model = pipeline.named_steps.get("model", None)

            # ================================================================
            # Get feature names after preprocessing
            # ================================================================
            # After one-hot encoding, categorical features become multiple binary features
            # get_feature_names returns the full list of transformed feature names
            feature_names = []
            try:
                feature_names = get_feature_names(preprocessor)
            except Exception:
                # Fallback: use original feature names (may not match after preprocessing)
                feature_names = list(X_test.columns)

            # ================================================================
            # Sample data for SHAP computation
            # ================================================================
            # SHAP can be slow on large datasets
            # We sample a subset for computational efficiency
            shap_sample_n = min(int(shap_sample), len(X_test))
            X_shap_sample = X_test.sample(n=shap_sample_n, random_state=random_state)
            
            print(f"Using {shap_sample_n} samples for SHAP computation")

            # ================================================================
            # Transform features to model input space
            # ================================================================
            # SHAP needs to operate on the transformed features (after preprocessing)
            if preprocessor is not None:
                X_shap_transformed = preprocessor.transform(X_shap_sample)
            else:
                # No preprocessing; use raw features
                X_shap_transformed = X_shap_sample.values

            # ================================================================
            # Create SHAP explainer
            # ================================================================
            # TreeExplainer is optimized for tree-based models (XGBoost, RandomForest)
            # It's much faster than KernelExplainer while being exact for trees
            explainer = None
            try:
                # Check if model is XGBoost (has get_booster method)
                if hasattr(model, "get_booster") or hasattr(model, "booster"):
                    explainer = shap.TreeExplainer(model)
                else:
                    # Fallback to generic explainer
                    explainer = shap.Explainer(model, X_shap_transformed)
            except Exception:
                # If TreeExplainer fails, use generic Explainer
                explainer = shap.Explainer(model, X_shap_transformed)

            # ================================================================
            # Compute SHAP values
            # ================================================================
            # SHAP values explain how much each feature contributed to prediction
            # Positive SHAP = pushes prediction higher (toward default)
            # Negative SHAP = pushes prediction lower (toward no default)
            shap_values = explainer.shap_values(X_shap_transformed)
            
            # ================================================================
            # Generate SHAP summary plot
            # ================================================================
            
            # ================================================================
            # Generate SHAP summary plot
            # ================================================================
            # Summary plot shows:
            # - Which features are most important overall
            # - How each feature impacts predictions (positive/negative)
            # - Distribution of feature values
            
            plt.figure(figsize=(8, 6))
            try:
                # Try to create standard SHAP summary plot
                # show=False prevents immediate display, allows us to save
                shap.summary_plot(
                    shap_values, 
                    X_shap_transformed, 
                    feature_names=feature_names, 
                    show=False
                )
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "shap_summary.png"), dpi=150)
                plt.close()
                print(f"✓ Saved SHAP summary plot")
            except Exception:
                # Fallback: Create bar plot of mean absolute SHAP values
                # This shows overall feature importance without direction
                importances = np.abs(shap_values).mean(axis=0)
                
                # Get top 30 most important features
                idx = np.argsort(importances)[::-1][:30]
                feat = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
                vals = importances[idx]
                
                # Create horizontal bar plot
                plt.figure(figsize=(8, 6))
                plt.barh(range(len(vals))[::-1], vals, color='steelblue')
                plt.yticks(range(len(vals))[::-1], feat)
                plt.xlabel("Mean |SHAP value|")
                plt.title("Feature Importance (Top 30)")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "shap_summary_bar.png"), dpi=150)
                plt.close()
                print(f"✓ Saved SHAP bar plot (fallback)")

            # Mark SHAP as successful
            shap_info["shap_available"] = True
            shap_info["shap_sample_used"] = shap_sample_n
            
        except Exception as e:
            # SHAP computation failed; record error but continue evaluation
            print(f"Warning: SHAP computation failed: {str(e)}")
            shap_info["error"] = str(e)
    else:
        # SHAP package not installed
        print(f"\nNote: SHAP package not installed, skipping explainability plots")
        shap_info["note"] = "shap not installed"

    # ========================================================================
    # STEP 11: CREATE PREDICTIONS SAMPLE FILE
    # ========================================================================
    # Save test set with predictions for manual inspection
    # Useful for debugging and understanding model behavior on specific cases
    
    print(f"\nCreating predictions sample file...")
    report_df = X_test.copy().reset_index(drop=True)
    report_df["_y_true"] = y_test.reset_index(drop=True)
    report_df["_y_proba"] = y_proba
    report_df["_y_pred"] = y_pred
    report_df.to_csv(os.path.join(out_dir, "predictions_sample.csv"), index=False)
    print(f"✓ Saved predictions to {out_dir}/predictions_sample.csv")

    # ========================================================================
    # STEP 12: SAVE EVALUATION SUMMARY
    # ========================================================================
    # Compile all results into single dictionary
    results = {
        "metrics": metrics,      # All computed metrics
        "plots_dir": plots_dir,  # Where plots were saved
        "shap": shap_info        # SHAP computation status
    }
    
    # Save as JSON for programmatic access
    summary_path = os.path.join(out_dir, "evaluation_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)
    
    print(f"✓ Saved evaluation summary to {summary_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nAll artifacts saved to: {out_dir}/")
    print(f"  - Metrics: evaluation_metrics.json, evaluation_metrics.csv")
    print(f"  - Plots: {plots_dir}/")
    print(f"  - Predictions: predictions_sample.csv")
    print(f"  - Summary: evaluation_summary.json")

    return results


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================
# This section handles command-line argument parsing for running the script
# directly from the terminal.

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluation configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - data_path: Path to input CSV file
            - pipeline_path: Path to trained pipeline.joblib
            - threshold_path: Path to threshold.json
            - out_dir: Output directory for artifacts
            - test_size: Test set fraction
            - random_state: Random seed
            - shap_sample: Max samples for SHAP
    
    Example:
        >>> args = parse_args()
        >>> print(args.data_path)
        data/raw/Loan_default.csv
    """
    parser = argparse.ArgumentParser(
        description="Evaluate trained credit risk pipeline with comprehensive metrics and plots"
    )
    
    # Required arguments
    parser.add_argument(
        "--data-path", 
        type=str, 
        required=True, 
        help="Path to raw CSV file with loan data"
    )
    
    # Optional model paths
    parser.add_argument(
        "--pipeline-path", 
        type=str, 
        default="models/pipeline.joblib", 
        help="Path to trained pipeline (default: models/pipeline.joblib)"
    )
    parser.add_argument(
        "--threshold-path", 
        type=str, 
        default="models/threshold.json", 
        help="Path to optimized threshold JSON (default: models/threshold.json)"
    )
    
    # Output configuration
    parser.add_argument(
        "--out-dir", 
        type=str, 
        default="artifacts", 
        help="Output directory for plots and metrics (default: artifacts)"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2, 
        help="Test set fraction (default: 0.2)"
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--shap-sample", 
        type=int, 
        default=1000, 
        help="Max samples for SHAP computation (default: 1000)"
    )
    
    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
# This block runs when the script is executed directly (not imported)

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Run evaluation pipeline
    results = evaluate_pipeline(
        data_path=args.data_path,
        pipeline_path=args.pipeline_path,
        threshold_path=args.threshold_path,
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        shap_sample=args.shap_sample,
    )
    
    # Print summary to console
    print("\n" + "="*60)
    print("METRICS SUMMARY")
    print("="*60)
    print(json.dumps(results["metrics"], indent=2))
    
    # Print SHAP status
    if results["shap"].get("shap_available"):
        print(f"\n✓ SHAP explanations generated using {results['shap']['shap_sample_used']} samples")
    else:
        print(f"\n⚠ SHAP not available — check artifacts for details")

