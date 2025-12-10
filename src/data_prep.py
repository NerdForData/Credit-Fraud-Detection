# src/data_prep.py
"""
Data preprocessing utilities for credit-risk-ml-system.

Usage:
    from src.data_prep import (
        get_feature_lists,
        get_preprocessor,
        fit_preprocessor,
        transform_with_preprocessor,
        save_preprocessor,
        load_preprocessor,
        prepare_features_and_target
    )
"""

from typing import List, Tuple, Dict, Any # for type hints
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib # for saving/loading models
import os

# ---------------------------
# Feature lists (dataset-specific)
# ---------------------------

def get_feature_lists() -> Dict[str, List[str]]:
    """
    Return numeric and categorical feature lists for this dataset.
    Update here if you add/remove features.
    """
    numeric_features = [
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "MonthsEmployed",
        "NumCreditLines",
        "InterestRate",
        "LoanTerm",
        "DTIRatio",
    ]

    # We treat LoanID as an identifier and drop it before training/inference.
    categorical_features = [
        "Education",
        "EmploymentType",
        "MaritalStatus",
        "LoanPurpose",
    ]

    binary_flags = [
        "HasMortgage",
        "HasDependents",
        "HasCoSigner",
    ]

    return {
        "numeric": numeric_features,
        "categorical": categorical_features,
        "binary": binary_flags,
        "id_column": "LoanID",
        "target": "Default",
    }


# ---------------------------
# Binary mapping helper
# ---------------------------

_BINARY_MAP = {"Yes": 1, "No": 0, "Y": 1, "N": 0, True: 1, False: 0, 1: 1, 0: 0}

def map_binary_flags(df: pd.DataFrame, binary_cols: List[str]) -> pd.DataFrame:
    """
    Convert common Yes/No/True/False representations into 1/0.
    Works in-place on a copy and returns the converted DataFrame.
    """
    df = df.copy()
    for col in binary_cols:
        if col not in df.columns:
            continue
        # Map common textual and boolean values to 0/1. Leave other values unchanged.
        df[col] = df[col].map(_BINARY_MAP).fillna(df[col])
        # If still object dtype (unexpected categories), attempt simple conversion
        if df[col].dtype == "object":
            # Try lower-case checks for 'yes'/'no'
            df[col] = df[col].str.strip().str.lower().map({"yes": 1, "no": 0}).fillna(df[col])
        # Finally ensure numeric dtype when possible
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass
    return df


# ---------------------------
# Preprocessor builder
# ---------------------------

def get_preprocessor() -> ColumnTransformer:
    """
    Build and return a ColumnTransformer handling numeric, categorical and binary columns.
    - numeric: impute (median) -> standard scale
    - categorical: impute (constant 'missing') -> one-hot (ignore unknowns)
    Binary flags should be converted beforehand (map_binary_flags).
    """
    features = get_feature_lists()
    numeric = features["numeric"]
    categorical = features["categorical"]

    # Numeric pipeline: impute median (robust) + standard scaling
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute missing string -> one-hot encode
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric),
            ("cat", categorical_transformer, categorical),
            # NOTE: binary flags are expected to be numeric (0/1) already; they will be passed separately.
        ],
        remainder="passthrough",  # passthrough allows binary columns to flow through (we'll select them)
        sparse_threshold=0.3,
    )

    return preprocessor


# ---------------------------
# Fit/transform helpers
# ---------------------------

def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Input: raw dataframe (as read from CSV)
    Returns: (X_df, y_series)
    - Drops LoanID
    - Converts binary flags to 0/1
    - Separates target column Default (y)
    - Leaves other columns intact (X_df)
    """
    features = get_feature_lists()
    id_col = features["id_column"]
    target_col = features["target"]
    binary_cols = features["binary"]

    df = df.copy()

    # Drop ID column if present
    if id_col in df.columns:
        df = df.drop(columns=[id_col])

    # Map binary flags to numeric 0/1 (in-place on the copy)
    df = map_binary_flags(df, binary_cols)

    # If target exists, separate it
    if target_col in df.columns:
        y = df[target_col].astype(int)
        X = df.drop(columns=[target_col])
    else:
        y = pd.Series(dtype=int)
        X = df

    return X, y


def fit_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, np.ndarray]:
    """
    Fit the preprocessor on the provided dataframe and return:
      - fitted preprocessor
      - transformed numpy array (X_transformed)
    Notes:
      - The caller should call prepare_features_and_target before passing df.
      - Binary flags must be numeric in X (they will be passed through by remainder='passthrough').
    """
    preprocessor = get_preprocessor()
    X, _ = prepare_features_and_target(df)

    # Ensure binary columns are positioned / present for passthrough:
    # ColumnTransformer with remainder='passthrough' will append any columns not listed in transformers
    # We rely on the numeric + categorical columns being present; binary cols should be in X.
    X_transformed = preprocessor.fit_transform(X)
    return preprocessor, X_transformed


def transform_with_preprocessor(preprocessor: ColumnTransformer, df: pd.DataFrame) -> np.ndarray:
    """
    Apply a fitted preprocessor to new dataframe and return transformed array.
    Use this in inference or after loading the saved preprocessor.
    """
    X, _ = prepare_features_and_target(df)
    X_transformed = preprocessor.transform(X)
    return X_transformed


# ---------------------------
# Save / Load preprocessor
# ---------------------------

def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    """
    Save fitted preprocessor to disk using joblib.
    Example: save_preprocessor(preproc, 'models/preprocessor.joblib')
    """
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str) -> ColumnTransformer:
    """
    Load a saved preprocessor from disk.
    """
    return joblib.load(path)


# ---------------------------
# Small utility: get feature names after preprocessing
# ---------------------------

def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Return a list of feature names after the ColumnTransformer transformation.
    Useful for SHAP/feature importance mapping.

    Note: Because remainder='passthrough' is used, the ordering is:
      - transformed numeric features (original numeric names)
      - one-hot encoded categorical features (expanded names)
      - passthrough columns (binary flags and any others not transformer-specified)

    This function handles sklearn >=1.0 conventions.
    """
    features = get_feature_lists()
    numeric = features["numeric"]
    categorical = features["categorical"]
    binary = features["binary"]

    feature_names: List[str] = []

    # Numeric names (they remain the same)
    feature_names.extend(numeric)

    # Categorical - expand onehot names
    try:
        # Access the onehot encoder inside the ColumnTransformer
        cat_transformer = None
        for name, trans, cols in preprocessor.transformers_:
            if name == "cat":
                # pipeline: SimpleImputer -> OneHotEncoder
                pipeline = trans
                onehot = pipeline.named_steps["onehot"]
                cat_transformer = (onehot, cols)
                break

        if cat_transformer is not None:
            onehot, cat_cols = cat_transformer
            # get categories for each feature
            categories = onehot.categories_
            for col, cats in zip(cat_cols, categories):
                # sanitize category names to strings and append
                names = [f"{col}__{str(c)}" for c in cats]
                feature_names.extend(names)
    except Exception:
        # If anything fails, we still fall back to a safe return
        pass

    # Finally, passthrough columns (we expect binary flags here)
    feature_names.extend(binary)

    return feature_names
