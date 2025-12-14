# src/train.py
"""
Model Training Pipeline for Credit Risk Prediction System.

This script implements the complete training workflow including:
- Data loading and preprocessing
- XGBoost classifier training with optimal hyperparameters
- Stratified K-Fold cross-validation for robust evaluation
- Threshold optimization to maximize F1 score
- Model persistence for production deployment

The training process addresses class imbalance using scale_pos_weight and
optimizes the classification threshold separately from model training to
achieve the best precision-recall tradeoff.

Key Features:
    - Stratified CV ensures balanced class distribution in each fold
    - Threshold tuning finds optimal cutoff point (not fixed at 0.5)
    - Class imbalance handling through XGBoost's scale_pos_weight
    - Comprehensive metrics tracking across all folds
    - Final model trained on full dataset for maximum performance

Usage:
    # Basic usage with default parameters
    python -m src.train --data-path data/raw/Loan_default.csv
    
    # Custom hyperparameters
    python -m src.train --data-path data/raw/Loan_default.csv \
        --n-estimators 300 --max-depth 8 --learning-rate 0.05
    
Output Files:
    - models/pipeline.joblib: Complete sklearn pipeline (preprocessor + model)
    - models/threshold.json: Optimized classification threshold
    - artifacts/metrics.csv: Cross-validation metrics for each fold

Example:
    >>> from src.train import run_training
    >>> results = run_training('data/raw/Loan_default.csv')
    >>> print(f"Best threshold: {results['threshold']:.4f}")
"""

# Standard library imports
import argparse
import json
import os
from typing import Dict, List, Tuple

# Third-party imports
import joblib
import numpy as np
import pandas as pd

# Scikit-learn imports
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

# XGBoost
from xgboost import XGBClassifier

# Local imports
from src.data_prep import get_preprocessor, prepare_features_and_target, get_feature_names



# ============================================================================
# THRESHOLD OPTIMIZATION FUNCTIONS
# ============================================================================
# These functions handle the critical task of finding the optimal classification
# threshold that maximizes F1 score. Unlike standard ML where threshold=0.5,
# we tune this separately to achieve the best precision-recall tradeoff.

def compute_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict]:
    """
    Find the optimal classification threshold that maximizes F1 score.
    
    This function evaluates all possible thresholds from the precision-recall
    curve and selects the one that gives the best F1 score. F1 is the harmonic
    mean of precision and recall, providing a balanced measure when we care
    equally about false positives and false negatives.
    
    Mathematical Foundation:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
        Precision = TP / (TP + FP)  - How many predicted defaults are correct?
        Recall = TP / (TP + FN)     - How many actual defaults did we catch?
    
    Why not use default 0.5 threshold?
    - In imbalanced datasets, 0.5 is rarely optimal
    - Business requirements may value precision over recall (or vice versa)
    - F1 maximization finds the sweet spot automatically
    
    Args:
        y_true (np.ndarray): Ground truth binary labels (0=no default, 1=default).
            Shape: (n_samples,)
        y_proba (np.ndarray): Predicted probabilities for the positive class (default).
            Shape: (n_samples,)
            Values should be in range [0, 1]
    
    Returns:
        Tuple[float, Dict]: A tuple containing:
            - best_threshold (float): Optimal threshold value that maximizes F1
            - best_metrics (Dict): Metrics at this threshold with keys:
                * 'precision': Precision at best threshold
                * 'recall': Recall at best threshold
                * 'f1': F1 score at best threshold
    
    Example:
        >>> y_true = np.array([0, 0, 1, 1, 1])
        >>> y_proba = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
        >>> threshold, metrics = compute_best_threshold(y_true, y_proba)
        >>> print(f"Best threshold: {threshold:.3f}, F1: {metrics['f1']:.3f}")
        Best threshold: 0.350, F1: 0.857
        
    Notes:
        - If multiple thresholds have the same F1, returns the first one
        - Uses 1e-12 epsilon to avoid division by zero
        - Fallback to 0.5 threshold if precision-recall curve is empty
        - NaN values in F1 scores are handled with np.nanargmax
    
    See Also:
        sklearn.metrics.precision_recall_curve: Generates P-R curve points
        evaluate_at_threshold: Uses this threshold to compute full metrics
    """
    # Generate precision-recall curve with all possible thresholds
    # Returns: precision[i], recall[i] at threshold=thresholds[i]
    # Note: thresholds has one fewer element than precision/recall
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Compute F1 score at each threshold
    # We slice precision/recall [:-1] to match thresholds length
    # Add small epsilon (1e-12) to denominator to prevent division by zero
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    
    # Handle edge case: empty F1 scores array
    if len(f1_scores) == 0:
        # Fallback to default threshold
        best_idx = 0
        best_threshold = 0.5
        best_metrics = {
            "precision": precision[0], 
            "recall": recall[0], 
            "f1": f1_scores[0] if len(f1_scores) > 0 else 0.0
        }
    else:
        # Find index with maximum F1 score (ignoring NaN values)
        best_idx = int(np.nanargmax(f1_scores))
        
        # Extract the threshold at this index
        best_threshold = thresholds[best_idx]
        
        # Build metrics dictionary at best threshold
        best_metrics = {
            "precision": float(precision[best_idx]),
            "recall": float(recall[best_idx]),
            "f1": float(f1_scores[best_idx]),
        }
    
    return float(best_threshold), best_metrics





def evaluate_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict:
    """
    Compute comprehensive classification metrics at a specific threshold.
    
    This function converts predicted probabilities into binary predictions using
    the provided threshold, then calculates precision, recall, F1 score, and
    confusion matrix. Used during cross-validation to evaluate model performance.
    
    Decision Rule:
        prediction = 1 (default) if probability >= threshold, else 0
    
    Args:
        y_true (np.ndarray): Ground truth binary labels (0 or 1).
            Shape: (n_samples,)
        y_proba (np.ndarray): Predicted probabilities for positive class.
            Shape: (n_samples,)
            Values in range [0, 1]
        threshold (float): Classification cutoff point.
            Typically in range [0.3, 0.7] after optimization
    
    Returns:
        Dict: Metrics dictionary with keys:
            - 'precision' (float): TP / (TP + FP)
            - 'recall' (float): TP / (TP + FN)
            - 'f1' (float): Harmonic mean of precision and recall
            - 'confusion_matrix' (List[List[int]]): 2x2 confusion matrix
                [[TN, FP],
                 [FN, TP]]
    
    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_proba = np.array([0.2, 0.8, 0.6, 0.3, 0.9])
        >>> metrics = evaluate_at_threshold(y_true, y_proba, threshold=0.5)
        >>> print(f"Precision: {metrics['precision']:.3f}")
        >>> print(f"Confusion Matrix: {metrics['confusion_matrix']}")
        
    Notes:
        - Uses zero_division=0 to handle undefined metrics gracefully
        - Confusion matrix format: [[TN, FP], [FN, TP]]
        - All metrics returned as Python floats (JSON serializable)
    """
    # Convert probabilities to binary predictions using threshold
    # If probability >= threshold, predict 1 (default), else 0 (no default)
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate precision: What fraction of predicted defaults are correct?
    # zero_division=0 means if no positive predictions, precision = 0
    p = precision_score(y_true, y_pred, zero_division=0)
    
    # Calculate recall: What fraction of actual defaults did we catch?
    # zero_division=0 means if no actual positives, recall = 0
    r = recall_score(y_true, y_pred, zero_division=0)
    
    # Calculate F1: Harmonic mean balancing precision and recall
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Generate confusion matrix
    # [[True Negatives,  False Positives],
    #  [False Negatives, True Positives]]
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "precision": float(p), 
        "recall": float(r), 
        "f1": float(f1), 
        "confusion_matrix": cm.tolist()  # Convert to list for JSON serialization
    }


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
# This section contains the core training workflow with cross-validation,
# threshold optimization, and model persistence.

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
    """
    Complete model training pipeline with cross-validation and threshold tuning.
    
    This function orchestrates the entire training workflow:
    1. Load and prepare data
    2. Calculate class imbalance ratio for XGBoost
    3. Build preprocessing + XGBoost pipeline
    4. Run stratified K-fold cross-validation
    5. Optimize classification threshold per fold
    6. Train final model on full dataset
    7. Save model, threshold, and metrics
    
    Training Strategy:
        - Stratified K-Fold ensures each fold has similar class distribution
        - XGBoost's scale_pos_weight handles class imbalance automatically
        - Threshold tuning finds optimal F1 cutoff point per fold
        - Final threshold is median of all fold thresholds (robust to outliers)
        - Final model trained on 100% of data for maximum performance
    
    Class Imbalance Handling:
        scale_pos_weight = count(negative) / count(positive)
        This makes XGBoost penalize misclassifying minority class more heavily
    
    Args:
        data_path (str): Path to raw CSV file with loan data.
            Must contain all required feature columns + 'Status' target
        model_out_path (str): Where to save trained pipeline.
            Default: "models/pipeline.joblib"
        threshold_out_path (str): Where to save optimized threshold JSON.
            Default: "models/threshold.json"
        metrics_out_path (str): Where to save CV metrics CSV.
            Default: "artifacts/metrics.csv"
        n_splits (int): Number of cross-validation folds.
            Default: 5 (80% train, 20% validation per fold)
        random_state (int): Random seed for reproducibility.
            Default: 42
        n_estimators (int): Number of boosting rounds for XGBoost.
            Default: 200 (balance between performance and training time)
        max_depth (int): Maximum tree depth for XGBoost.
            Default: 6 (prevents overfitting)
        learning_rate (float): Step size shrinkage for XGBoost.
            Default: 0.1 (conservative learning)
    
    Returns:
        Dict: Training results with keys:
            - 'threshold' (float): Optimized classification threshold
            - 'cv_metrics' (List[Dict]): Metrics for each CV fold
            - 'model_path' (str): Path to saved model
            - 'threshold_path' (str): Path to saved threshold
    
    Side Effects:
        - Creates output directories if they don't exist
        - Saves pipeline.joblib to model_out_path
        - Saves threshold.json to threshold_out_path
        - Saves metrics.csv to metrics_out_path
        - Prints progress updates to console
    
    Example:
        >>> results = run_training(
        ...     data_path='data/raw/Loan_default.csv',
        ...     n_estimators=300,
        ...     max_depth=8
        ... )
        >>> print(f"Model saved to: {results['model_path']}")
        >>> print(f"Best threshold: {results['threshold']:.4f}")
        
    Notes:
        - Requires at least 2 samples per class for stratified split
        - Metrics CSV includes fold-by-fold performance tracking
        - Final model uses all data; no holdout test set
        - Threshold optimization happens per-fold, then median is taken
    
    See Also:
        compute_best_threshold: Finds F1-optimal threshold
        evaluate_at_threshold: Computes metrics at given threshold
        get_preprocessor: Builds feature transformation pipeline
    """
    # Create output directories if they don't exist
    # Use "." as fallback for current directory if no dirname
    os.makedirs(os.path.dirname(model_out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(threshold_out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metrics_out_path) or "artifacts", exist_ok=True)

    # ========================================================================
    # STEP 1: DATA LOADING AND PREPARATION
    # ========================================================================
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Prepare features (X) and target (y)
    # This handles binary mapping, feature selection, etc.
    X, y = prepare_features_and_target(df)

    # Sanity check: ensure features and labels are aligned
    assert len(X) == len(y), "Features and target lengths must match."
    print(f"Loaded data: {len(X)} rows, {X.shape[1]} features")

    # ========================================================================
    # STEP 2: CLASS IMBALANCE ANALYSIS
    # ========================================================================
    # Calculate class distribution to configure XGBoost's scale_pos_weight
    # This parameter tells XGBoost how much more to weight the minority class
    neg = int((y == 0).sum())  # Count of non-defaults
    pos = int((y == 1).sum())  # Count of defaults
    
    # scale_pos_weight = negative_count / positive_count
    # Example: If 900 non-defaults and 100 defaults, ratio = 9.0
    # This makes XGBoost treat each default as 9x more important
    scale_pos_weight = neg / (pos + 1e-9)  # Add epsilon to avoid division by zero
    
    print(f"Positive / Negative counts: {pos} / {neg}  -> scale_pos_weight={scale_pos_weight:.2f}")

    # ========================================================================
    # STEP 3: BUILD PIPELINE
    # ========================================================================
    # Get preprocessing pipeline (handles numeric scaling, categorical encoding)
    preprocessor = get_preprocessor()
    
    # Configure XGBoost classifier with specified hyperparameters
    xgb = XGBClassifier(
        n_estimators=n_estimators,      # Number of boosting rounds (trees)
        max_depth=max_depth,              # Maximum tree depth (prevents overfitting)
        learning_rate=learning_rate,      # Step size shrinkage (eta)
        use_label_encoder=False,          # Disable deprecated label encoder
        eval_metric="logloss",            # Use log loss for binary classification
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        random_state=random_state,        # Reproducibility
        verbosity=0,                      # Suppress training logs
    )

    # Combine preprocessor and model into a single pipeline
    # This ensures preprocessing is applied consistently in training and inference
    pipeline = Pipeline([
        ("preprocessor", preprocessor),  # Step 1: Transform features
        ("model", xgb),                  # Step 2: XGBoost classification
    ])

    # ========================================================================
    # STEP 4: STRATIFIED K-FOLD CROSS-VALIDATION
    # ========================================================================
    # Why Stratified K-Fold?
    # - Regular K-Fold might create folds with different class ratios
    # - Stratified ensures each fold has similar positive/negative ratio
    # - Critical for imbalanced datasets to get reliable validation metrics
    
    skf = StratifiedKFold(
        n_splits=n_splits,           # Number of folds (e.g., 5 = 80/20 split)
        shuffle=True,                 # Randomize before splitting
        random_state=random_state     # Reproducible splits
    )

    # Storage for fold-level results
    fold_results: List[Dict] = []      # Detailed metrics per fold
    best_thresholds: List[float] = []  # Optimal threshold per fold

    # ========================================================================
    # CROSS-VALIDATION LOOP
    # ========================================================================
    # For each fold:
    # 1. Train on 80% of data (train_idx)
    # 2. Validate on 20% of data (val_idx)
    # 3. Find best threshold on validation set
    # 4. Record all metrics
    
    print(f"\nStarting {n_splits}-fold cross-validation...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n{'='*60}")
        print(f"Training fold {fold_idx}/{n_splits}")
        print(f"{'='*60}")
        
        # Split data into training and validation sets for this fold
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        print(f"Train set: {len(X_train)} samples | Val set: {len(X_val)} samples")

        # Train pipeline on this fold's training data
        # Pipeline automatically applies preprocessing before XGBoost
        pipeline.fit(X_train, y_train)

        # Get predicted probabilities for positive class (default) on validation set
        # predict_proba returns shape (n_samples, 2): [prob_class_0, prob_class_1]
        # We take [:, 1] to get only the positive class probabilities
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]

        # ====================================================================
        # THRESHOLD-INDEPENDENT METRICS
        # ====================================================================
        # These metrics don't require a threshold; they use probabilities directly
        
        # ROC-AUC: Area under ROC curve (FPR vs TPR)
        # Measures how well model separates classes across all thresholds
        # Range: 0.5 (random) to 1.0 (perfect)
        roc = float(roc_auc_score(y_val, y_val_proba))
        
        # PR-AUC: Area under precision-recall curve
        # More informative than ROC-AUC for imbalanced datasets
        # Focuses on performance on positive class
        pr_auc = float(average_precision_score(y_val, y_val_proba))

        # ====================================================================
        # THRESHOLD OPTIMIZATION
        # ====================================================================
        # Find the threshold that maximizes F1 score on validation set
        best_thr, best_metrics = compute_best_threshold(y_val.values, y_val_proba)
        
        # Evaluate model performance at this optimal threshold
        eval_metrics = evaluate_at_threshold(y_val.values, y_val_proba, best_thr)

        # Build comprehensive results dictionary for this fold
        fold_result = {
            "fold": fold_idx,
            
            # Threshold-independent metrics
            "roc_auc": roc,
            "pr_auc": pr_auc,
            
            # Optimal threshold found
            "best_threshold": best_thr,
            
            # Metrics from compute_best_threshold (during PR curve scan)
            "best_precision_on_pr": best_metrics.get("precision"),
            "best_recall_on_pr": best_metrics.get("recall"),
            "best_f1_on_pr": best_metrics.get("f1"),
            
            # Metrics from evaluate_at_threshold (verification)
            "precision_at_best_threshold": eval_metrics["precision"],
            "recall_at_best_threshold": eval_metrics["recall"],
            "f1_at_best_threshold": eval_metrics["f1"],
            
            # Confusion matrix at optimal threshold
            "confusion_matrix": eval_metrics["confusion_matrix"],
        }
        
        # Store results
        fold_results.append(fold_result)
        best_thresholds.append(best_thr)

        # Print fold summary
        print(f"\nFold {fold_idx} Results:")
        print(f"  ROC-AUC: {roc:.4f}")
        print(f"  PR-AUC:  {pr_auc:.4f}")
        print(f"  Best Threshold: {best_thr:.4f}")
        print(f"  F1 Score: {eval_metrics['f1']:.4f}")
        print(f"  Precision: {eval_metrics['precision']:.4f}")
        print(f"  Recall: {eval_metrics['recall']:.4f}")

    # ========================================================================
    # STEP 5: SAVE CROSS-VALIDATION METRICS
    # ========================================================================
    print(f"\n{'='*60}")
    print("Cross-Validation Complete!")
    print(f"{'='*60}")
    
    # Convert fold results to DataFrame for easy analysis
    metrics_df = pd.DataFrame(fold_results)
    metrics_df.to_csv(metrics_out_path, index=False)
    print(f"\nSaved CV fold metrics to: {metrics_out_path}")
    
    # Print summary statistics across all folds
    print(f"\nCross-Validation Summary (mean ± std):")
    print(f"  ROC-AUC: {metrics_df['roc_auc'].mean():.4f} ± {metrics_df['roc_auc'].std():.4f}")
    print(f"  PR-AUC:  {metrics_df['pr_auc'].mean():.4f} ± {metrics_df['pr_auc'].std():.4f}")
    print(f"  F1:      {metrics_df['f1_at_best_threshold'].mean():.4f} ± {metrics_df['f1_at_best_threshold'].std():.4f}")

    # ========================================================================
    # STEP 6: TRAIN FINAL MODEL ON FULL DATASET
    # ========================================================================
    # Why train on full data after CV?
    # - CV was for evaluation/validation only
    # - Using 100% of data maximizes model performance in production
    # - We already have reliable performance estimates from CV
    
    print(f"\n{'='*60}")
    print("Training final model on full dataset...")
    print(f"{'='*60}")
    pipeline.fit(X, y)
    print("✓ Final model training complete!")

    # ========================================================================
    # STEP 7: SELECT FINAL THRESHOLD
    # ========================================================================
    # Choose final threshold as median of best fold thresholds
    # Why median instead of mean?
    # - More robust to outlier thresholds from any single fold
    # - Represents typical optimal threshold across CV folds
    # - Less sensitive to extreme values
    
    if len(best_thresholds) > 0:
        final_threshold = float(np.median(best_thresholds))
        print(f"\nFinal threshold (median of {len(best_thresholds)} folds): {final_threshold:.4f}")
        print(f"  Threshold range: [{min(best_thresholds):.4f}, {max(best_thresholds):.4f}]")
    else:
        # Fallback if no thresholds were computed (shouldn't happen in normal flow)
        final_threshold = 0.5
        print(f"\nWarning: No thresholds computed, using default: {final_threshold}")

    # ========================================================================
    # STEP 8: SAVE MODEL AND THRESHOLD
    # ========================================================================
    # Save trained pipeline to disk
    # This includes both the preprocessor and the fitted XGBoost model
    joblib.dump(pipeline, model_out_path)
    print(f"\n✓ Saved final pipeline to: {model_out_path}")
    
    # Save threshold as JSON for easy loading in production
    with open(threshold_out_path, "w") as fh:
        json.dump({"threshold": final_threshold}, fh)
    print(f"✓ Saved optimal threshold to: {threshold_out_path}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nModel artifacts saved:")
    print(f"  - Pipeline: {model_out_path}")
    print(f"  - Threshold: {threshold_out_path}")
    print(f"  - CV Metrics: {metrics_out_path}")
    
    print(f"\nFinal Configuration:")
    print(f"  - Threshold: {final_threshold:.4f}")
    print(f"  - CV Folds: {n_splits}")
    print(f"  - XGBoost n_estimators: {n_estimators}")
    print(f"  - XGBoost max_depth: {max_depth}")
    print(f"  - XGBoost learning_rate: {learning_rate}")

    # Return results dictionary for programmatic access
    return {
        "metrics_df": metrics_df,           # DataFrame with all fold metrics
        "pipeline_path": model_out_path,    # Path to saved model
        "threshold": final_threshold,       # Optimized threshold
    }


# ============================================================================
# COMMAND-LINE INTERFACE
# ============================================================================
# This section handles command-line argument parsing for running the script
# directly from the terminal.

def parse_args():
    """
    Parse command-line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments with attributes:
            - data_path: Path to input CSV file
            - model_out: Path to save pipeline.joblib
            - threshold_out: Path to save threshold.json
            - metrics_out: Path to save metrics.csv
            - n_splits: Number of CV folds
            - n_estimators: XGBoost trees
            - max_depth: XGBoost tree depth
            - learning_rate: XGBoost learning rate
            - random_state: Random seed
    
    Example:
        >>> args = parse_args()
        >>> print(args.data_path)
        data/raw/Loan_default.csv
    """
    parser = argparse.ArgumentParser(
        description="Train credit-risk ML pipeline with cross-validation and threshold tuning"
    )
    
    # Required arguments
    parser.add_argument(
        "--data-path", 
        type=str, 
        required=True, 
        help="Path to raw CSV file with loan data"
    )
    
    # Optional output paths
    parser.add_argument(
        "--model-out", 
        type=str, 
        default="models/pipeline.joblib", 
        help="Output path for trained pipeline (default: models/pipeline.joblib)"
    )
    parser.add_argument(
        "--threshold-out", 
        type=str, 
        default="models/threshold.json", 
        help="Output path for optimized threshold JSON (default: models/threshold.json)"
    )
    parser.add_argument(
        "--metrics-out", 
        type=str, 
        default="artifacts/metrics.csv", 
        help="Output path for CV metrics CSV (default: artifacts/metrics.csv)"
    )
    
    # Cross-validation configuration
    parser.add_argument(
        "--n-splits", 
        type=int, 
        default=5, 
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility (default: 42)"
    )
    
    # XGBoost hyperparameters
    parser.add_argument(
        "--n-estimators", 
        type=int, 
        default=200, 
        help="Number of boosting rounds (default: 200)"
    )
    parser.add_argument(
        "--max-depth", 
        type=int, 
        default=6, 
        help="Maximum tree depth (default: 6)"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=0.1, 
        help="Learning rate / eta (default: 0.1)"
    )
    
    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
# This block runs when the script is executed directly (not imported)

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    
    # Run training pipeline with parsed arguments
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

