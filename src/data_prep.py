# src/data_prep.py
"""
Data Preprocessing Module for Credit Risk Prediction System.

This module handles all data preprocessing tasks including:
- Feature type identification (numeric, categorical, binary)
- Binary flag conversion (Yes/No → 1/0)
- Missing value imputation
- Feature scaling and encoding
- Preprocessing pipeline creation and management

The preprocessing pipeline ensures that raw loan application data is transformed
into a format suitable for the XGBoost model while handling edge cases like
missing values, unknown categories, and different data types.

Usage Example:
    from src.data_prep import (
        get_feature_lists,
        get_preprocessor,
        prepare_features_and_target
    )
    
    # Load raw data
    df = pd.read_csv('loan_data.csv')
    
    # Prepare features and target
    X, y = prepare_features_and_target(df)
    
    # Get preprocessor and transform data
    preprocessor = get_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
"""

# Standard library imports
import os
from typing import List, Tuple, Dict, Any

# Third-party imports
import pandas as pd
import numpy as np
import joblib

# Scikit-learn imports for preprocessing pipeline
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ============================================================================
# FEATURE CONFIGURATION
# ============================================================================
# This section defines which columns belong to each feature type.
# Update these lists if you add/remove features from your dataset.

def get_feature_lists() -> Dict[str, List[str]]:
    """
    Define and return all feature categories for the loan default dataset.
    
    This function organizes features into three categories:
    1. Numeric: Continuous or discrete numbers (scaled with StandardScaler)
    2. Categorical: Text categories (encoded with OneHotEncoder)
    3. Binary: Yes/No flags (converted to 1/0)
    
    Returns:
        Dict containing:
            - "numeric": List of numeric feature names
            - "categorical": List of categorical feature names  
            - "binary": List of binary flag feature names
            - "id_column": Name of ID column to drop
            - "target": Name of target column (Default)
    
    Note:
        If you add new features to your dataset, update these lists accordingly.
    """
    # Numeric features: These will be imputed (median) and scaled (z-score)
    numeric_features = [
        "Age",              # Applicant's age in years
        "Income",           # Annual income in dollars
        "LoanAmount",       # Requested loan amount in dollars
        "CreditScore",      # Credit score (300-850 range)
        "MonthsEmployed",   # Duration at current job
        "NumCreditLines",   # Number of open credit lines
        "InterestRate",     # Loan interest rate percentage
        "LoanTerm",         # Loan duration in months
        "DTIRatio",         # Debt-to-Income ratio (0-1)
    ]

    # Categorical features: These will be one-hot encoded
    # Note: LoanID is treated as an identifier and dropped, not used for prediction
    categorical_features = [
        "Education",        # Education level (High School, Bachelor's, Master's, PhD)
        "EmploymentType",   # Type of employment (Full-time, Part-time, etc.)
        "MaritalStatus",    # Marital status (Single, Married, Divorced)
        "LoanPurpose",      # Purpose of loan (Home, Auto, Education, Business)
    ]

    # Binary features: Yes/No flags converted to 1/0
    binary_flags = [
        "HasMortgage",      # Whether applicant has a mortgage
        "HasDependents",    # Whether applicant has dependents
        "HasCoSigner",      # Whether loan has a co-signer
    ]

    return {
        "numeric": numeric_features,
        "categorical": categorical_features,
        "binary": binary_flags,
        "id_column": "LoanID",      # Identifier column to drop
        "target": "Default",         # Target variable (0=No Default, 1=Default)
    }


# ============================================================================
# BINARY FLAG CONVERSION
# ============================================================================
# Convert Yes/No, True/False, Y/N into 1/0 for machine learning models.

# Mapping dictionary for binary value conversion
# Handles multiple formats: Yes/No, Y/N, True/False, 1/0
_BINARY_MAP = {
    "Yes": 1, "No": 0,      # String format
    "Y": 1, "N": 0,          # Abbreviated format
    True: 1, False: 0,       # Boolean format
    1: 1, 0: 0               # Already numeric
}

def map_binary_flags(df: pd.DataFrame, binary_cols: List[str]) -> pd.DataFrame:
    """
    Convert binary flags (Yes/No, True/False) to numeric 1/0 format.
    
    This function handles multiple representations of binary data:
    - "Yes"/"No" → 1/0
    - "Y"/"N" → 1/0
    - True/False → 1/0
    - Case-insensitive handling
    
    Args:
        df: Input DataFrame with binary columns
        binary_cols: List of column names containing binary data
    
    Returns:
        DataFrame with binary columns converted to numeric 1/0
    
    Example:
        >>> df = pd.DataFrame({'HasMortgage': ['Yes', 'No', 'Yes']})
        >>> df = map_binary_flags(df, ['HasMortgage'])
        >>> df['HasMortgage']
        0    1
        1    0
        2    1
    
    Note:
        - Works on a copy of the DataFrame (doesn't modify original)
        - Handles missing values gracefully
        - Attempts case-insensitive conversion as fallback
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    for col in binary_cols:
        # Skip if column doesn't exist in DataFrame
        if col not in df.columns:
            continue
        
        # Step 1: Apply the mapping dictionary to convert common formats
        df[col] = df[col].map(_BINARY_MAP).fillna(df[col])
        
        # Step 2: Handle case-insensitive string values as fallback
        if df[col].dtype == "object":
            # Clean and convert lowercase 'yes'/'no' strings
            df[col] = (
                df[col]
                .str.strip()           # Remove whitespace
                .str.lower()           # Convert to lowercase
                .map({"yes": 1, "no": 0})  # Map to 1/0
                .fillna(df[col])       # Keep original if not yes/no
            )
        
        # Step 3: Ensure final dtype is numeric
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            # If conversion fails, keep as is
            pass
    
    return df


# ============================================================================
# PREPROCESSING PIPELINE BUILDER
# ============================================================================
# Create sklearn ColumnTransformer to handle different feature types.

def get_preprocessor() -> ColumnTransformer:
    """
    Build and return a complete preprocessing pipeline using ColumnTransformer.
    
    This pipeline handles three types of features differently:
    
    1. **Numeric Features** (Age, Income, CreditScore, etc.):
       - Impute missing values with median (robust to outliers)
       - Scale using StandardScaler (z-score normalization)
       - Result: Mean=0, StdDev=1
    
    2. **Categorical Features** (Education, EmploymentType, etc.):
       - Impute missing values with constant 'missing'
       - One-hot encode categories
       - Handle unknown categories gracefully (ignore_unknown)
       - Result: Binary columns for each category
    
    3. **Binary Features** (HasMortgage, HasDependents, HasCoSigner):
       - Should be converted to 1/0 BEFORE using this preprocessor
       - Passed through unchanged (remainder='passthrough')
    
    Returns:
        ColumnTransformer: Fitted preprocessing pipeline
    
    Pipeline Flow:
        Raw Data → [Numeric Transform] → [Categorical Transform] → [Binary Passthrough] → Model
    
    Note:
        - Binary flags must be converted to 0/1 using map_binary_flags() first
        - The pipeline is designed to be saved and reused for inference
        - Unknown categories in production are handled gracefully
    """
    # Get feature configurations
    features = get_feature_lists()
    numeric = features["numeric"]
    categorical = features["categorical"]

    # -----------------------------------------------------------------------
    # Pipeline 1: Numeric Features Transformation
    # -----------------------------------------------------------------------
    # Strategy: Median imputation (robust to outliers) + Standard scaling
    numeric_transformer = Pipeline(
        steps=[
            # Step 1: Fill missing values with median
            # Why median? More robust than mean for skewed distributions
            ("imputer", SimpleImputer(strategy="median")),
            
            # Step 2: Standardize features (z-score normalization)
            # Formula: (x - mean) / std_dev
            # Result: Mean=0, StdDev=1 for each feature
            ("scaler", StandardScaler()),
        ]
    )

    # -----------------------------------------------------------------------
    # Pipeline 2: Categorical Features Transformation
    # -----------------------------------------------------------------------
    # Strategy: Constant imputation + One-Hot encoding
    categorical_transformer = Pipeline(
        steps=[
            # Step 1: Fill missing values with constant 'missing'
            # This creates a separate category for missing values
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            
            # Step 2: One-Hot encode categorical variables
            # Creates binary (0/1) columns for each category
            # handle_unknown='ignore': New categories in production won't cause errors
            # sparse_output=False: Return dense array (easier to work with)
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    # -----------------------------------------------------------------------
    # Combine All Transformers
    # -----------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            # Apply numeric transformer to numeric columns
            ("num", numeric_transformer, numeric),
            
            # Apply categorical transformer to categorical columns
            ("cat", categorical_transformer, categorical),
            
            # Note: Binary columns (HasMortgage, etc.) are handled by remainder
        ],
        # remainder='passthrough': Let binary columns (already 0/1) pass through unchanged
        remainder="passthrough",
        
        # sparse_threshold: Use sparse matrix if >30% of values are zero
        sparse_threshold=0.3,
    )

    return preprocessor


# ============================================================================
# DATA PREPARATION FUNCTIONS
# ============================================================================
# Functions to prepare raw data for training and inference.

def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare raw loan data for machine learning by cleaning and separating features/target.
    
    This function performs several critical preprocessing steps:
    1. Removes the ID column (LoanID) - not useful for prediction
    2. Converts binary flags (Yes/No) to numeric (1/0)
    3. Separates features (X) from target variable (y)
    
    Args:
        df: Raw DataFrame as loaded from CSV file
            Expected columns: LoanID, Age, Income, ..., Default
    
    Returns:
        Tuple of:
            - X (DataFrame): Feature matrix without target column
            - y (Series): Target variable (0=No Default, 1=Default)
            
            If target column doesn't exist (inference mode):
                - y will be an empty Series
    
    Example:
        >>> df = pd.read_csv('loan_data.csv')
        >>> X, y = prepare_features_and_target(df)
        >>> print(X.shape)  # (n_samples, n_features)
        >>> print(y.shape)  # (n_samples,)
    
    Processing Steps:
        Input DataFrame
            ↓
        Remove LoanID (identifier, not predictive)
            ↓
        Convert HasMortgage, HasDependents, HasCoSigner (Yes/No → 1/0)
            ↓
        Separate Default column (target) from features
            ↓
        Return (X, y)
    """
    # Get feature configuration
    features = get_feature_lists()
    id_col = features["id_column"]        # LoanID
    target_col = features["target"]        # Default
    binary_cols = features["binary"]      # HasMortgage, HasDependents, HasCoSigner

    # Work on a copy to avoid modifying the original DataFrame
    df = df.copy()

    # -----------------------------------------------------------------------
    # Step 1: Drop ID column (LoanID)
    # -----------------------------------------------------------------------
    # LoanID is just an identifier and provides no predictive power
    if id_col in df.columns:
        df = df.drop(columns=[id_col])

    # -----------------------------------------------------------------------
    # Step 2: Convert binary flags to numeric 0/1
    # -----------------------------------------------------------------------
    # Convert Yes/No, True/False, Y/N to 1/0 for ML model
    df = map_binary_flags(df, binary_cols)

    # -----------------------------------------------------------------------
    # Step 3: Separate features (X) and target (y)
    # -----------------------------------------------------------------------
    if target_col in df.columns:
        # Training mode: Target column exists
        y = df[target_col].astype(int)     # Target: 0 or 1
        X = df.drop(columns=[target_col])   # Features: Everything except target
    else:
        # Inference mode: No target column (predicting on new data)
        y = pd.Series(dtype=int)            # Empty series
        X = df                               # All columns are features

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
