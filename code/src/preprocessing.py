import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean_data(file_path):
    """
    Loads dataset, handles missing values (marked as empty, '?' or NaN)
    using median imputation, and maps multi-class targets to binary.
    """
    df = pd.read_csv(file_path)
    
    # Clean string values, replace '?' with NaN and coerce to float
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({'?': np.nan, '': np.nan})
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Impute missing values with column median
    # Cleveland ca and thal columns have missing values
    for col in df.columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # Target variable: map target >= 1 (disease presence) to 1, normal (0) to 0
    if 'target' in df.columns:
        df['target'] = df['target'].apply(lambda x: 1 if x >= 1 else 0)
        
    return df

def cap_outliers_iqr(df, numerical_cols, factor=1.5):
    """
    Caps outliers in numerical columns using the Interquartile Range (IQR) method.
    This preserves dataset size (important for small datasets like ~300 records).
    """
    df_capped = df.copy()
    for col in numerical_cols:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df_capped[col] = np.clip(df_capped[col], lower_bound, upper_bound)
    return df_capped

def batch_compute_gasf(X):
    """
    Computes GAF (GASF) images for a batch of 1D vectors.
    Input shape: (N, T)
    Output shape: (N, T, T, 1)
    """
    N, T = X.shape
    gasf_batch = np.zeros((N, T, T, 1))
    for i in range(N):
        series = X[i]
        min_val = np.min(series)
        max_val = np.max(series)
        if max_val - min_val == 0:
            scaled = np.zeros_like(series)
        else:
            scaled = -1 + 2 * (series - min_val) / (max_val - min_val)
        scaled = np.clip(scaled, -1.0, 1.0)
        
        phi = np.arccos(scaled)
        cos_phi = scaled
        sin_phi = np.sin(phi)
        gasf_matrix = np.outer(cos_phi, cos_phi) - np.outer(sin_phi, sin_phi)
        gasf_batch[i, :, :, 0] = gasf_matrix
    return gasf_batch

class ClinicalRangeScaler:
    """
    Scales numerical clinical features to a [0, 1] range using fixed
    physiological boundaries rather than dataset-dependent statistics.
    This ensures out-of-sample consistency and physiological interpretability.
    """
    def __init__(self):
        # Clinical min and max physiological ranges
        self.bounds = {
            'age': (25.0, 85.0),
            'trestbps': (80.0, 200.0),
            'chol': (100.0, 400.0),
            'thalach': (60.0, 220.0),
            'oldpeak': (0.0, 6.0)
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_scaled = X.copy()
        if isinstance(X_scaled, pd.DataFrame):
            for col, (c_min, c_max) in self.bounds.items():
                if col in X_scaled.columns:
                    clipped = np.clip(X_scaled[col], c_min, c_max)
                    X_scaled[col] = (clipped - c_min) / (c_max - c_min)
        return X_scaled

    def fit_transform(self, X, y=None):
        return self.transform(X)

def preprocess_pipeline(file_path, numerical_cols, categorical_cols, apply_smote=True, random_state=42):
    """
    Complete preprocessing pipeline.
    1. Loads and imputes missing data.
    2. Scales numerical features using a custom ClinicalRangeScaler.
    3. One-hot encodes categorical columns.
    4. Creates mapping from original features to one-hot indices.
    5. Performs Stratified Train-Val-Test Split (70%-15%-15%).
    6. Applies SMOTE to balance the training split.
    7. Reshapes tabular data to (samples, features, 1) for Conv1D.
    """
    df = load_and_clean_data(file_path)
    
    # 1. Scale numerical features using clinical bounds
    scaler = ClinicalRangeScaler()
    scaled_numerical = scaler.fit_transform(df[numerical_cols])
    df_scaled_num = pd.DataFrame(scaled_numerical, columns=numerical_cols)
    
    # 2. Process categorical features
    # Convert categorical features to category type for dummy generation
    df_cat = df[categorical_cols].astype(str)
    
    # Generate one-hot encoded columns (dummy variables)
    df_encoded_cat = pd.get_dummies(df_cat, columns=categorical_cols, prefix=categorical_cols, dtype=float)
    
    # 3. Recombine scaled numerical and encoded categorical features
    X_processed = pd.concat([df_scaled_num, df_encoded_cat], axis=1)
    y = df['target'].copy()
    
    # 4. Generate original-to-processed feature mapping
    # This mapping is critical for XAI feature aggregation
    feature_mapping = {}
    processed_cols = list(X_processed.columns)
    
    # Numerical features map directly 1:1
    for col in numerical_cols:
        if col in processed_cols:
            feature_mapping[col] = [processed_cols.index(col)]
            
    # Categorical features map to all their respective one-hot dummy column indices
    for orig_col in categorical_cols:
        matched_cols = [col for col in processed_cols if col.startswith(f"{orig_col}_")]
        feature_mapping[orig_col] = [processed_cols.index(col) for col in matched_cols]
        
    # 5. Stratified Train-Validation-Test Split (70% Train, 10% Val, 20% Test)
    # First split off 20% for Test (leaving 80% for Train/Val)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_processed, y, test_size=0.20, stratify=y, random_state=random_state
    )
    # Then split Train/Val (10% of total is 12.5% of train_val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.125, stratify=y_train_val, random_state=random_state
    )
    
    X_train_arr, y_train_arr = X_train.values, y_train.values
    
    # 6. Apply SMOTE to the training set if requested
    smote_applied = False
    if apply_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=random_state)
            X_train_arr, y_train_arr = smote.fit_resample(X_train.values, y_train.values)
            print(f"SMOTE applied: Training set shape increased from {len(X_train)} to {len(X_train_arr)}.")
            smote_applied = True
        except ImportError:
            print("imblearn (SMOTE) not available. Training pipeline will fall back to using class weights.")
            
    # 7. Convert data to 2D GAF (GASF) images: (samples, features, features, 1)
    X_train_reshaped = batch_compute_gasf(X_train_arr)
    X_val_reshaped = batch_compute_gasf(X_val.values)
    X_test_reshaped = batch_compute_gasf(X_test.values)
    
    return {
        'X_train': X_train_reshaped,
        'y_train': y_train_arr,
        'X_val': X_val_reshaped,
        'y_val': y_val.values,
        'X_test': X_test_reshaped,
        'y_test': y_test.values,
        'scaler': scaler,
        'feature_names': processed_cols,
        'original_features': list(df.columns)[:-1],
        'feature_mapping': feature_mapping,
        'smote_applied': smote_applied,
        'X_test_indices': list(X_test.index),
        'X_train_1d': X_train_arr,
        'X_val_1d': X_val.values,
        'X_test_1d': X_test.values
    }
