"""
Data preprocessing utilities for customer churn prediction.

This module contains functions for data cleaning, encoding, scaling,
and feature engineering used in the churn prediction pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict, Any


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and testing datasets.
    
    Args:
        train_path: Path to training CSV file
        test_path: Path to testing CSV file
        
    Returns:
        Tuple of (train_df, test_df)
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"✅ Data loaded successfully:")
        print(f"   Training: {train_df.shape}")
        print(f"   Testing: {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        raise


def encode_binary_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Encode binary categorical features (Yes/No, True/False) to 1/0.
    
    Args:
        df: Input dataframe
        columns: List of column names to encode
        
    Returns:
        DataFrame with encoded features
    """
    df_encoded = df.copy()
    binary_map = {'Yes': 1, 'No': 0, True: 1, False: 0}
    
    for col in columns:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(binary_map)
            
    return df_encoded


def create_preprocessing_pipeline(numerical_cols: List[str], 
                                categorical_cols: List[str]) -> ColumnTransformer:
    """
    Create preprocessing pipeline for numerical and categorical features.
    
    Args:
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        
    Returns:
        Fitted ColumnTransformer pipeline
    """
    # Numerical preprocessing
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing  
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'  # Keep binary encoded columns
    )
    
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, 
                     numerical_cols: List[str],
                     categorical_cols: List[str],
                     remainder_cols: List[str]) -> List[str]:
    """
    Get feature names after preprocessing transformation.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        numerical_cols: Original numerical column names
        categorical_cols: Original categorical column names  
        remainder_cols: Columns passed through unchanged
        
    Returns:
        List of feature names after transformation
    """
    # Get categorical feature names after one-hot encoding
    cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
    
    # Combine all feature names
    feature_names = list(numerical_cols) + list(cat_features) + list(remainder_cols)
    
    return feature_names


def remove_correlated_features(X: pd.DataFrame, 
                             threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove highly correlated features to reduce multicollinearity.
    
    Args:
        X: Feature dataframe
        threshold: Correlation threshold for removal
        
    Returns:
        Tuple of (cleaned_dataframe, removed_columns)
    """
    correlation_matrix = X.corr().abs()
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation above threshold
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]
    
    X_cleaned = X.drop(columns=to_drop)
    
    print(f"🔧 Removed {len(to_drop)} highly correlated features: {to_drop}")
    
    return X_cleaned, to_drop


def preprocess_data(train_df: pd.DataFrame, 
                   test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Complete preprocessing pipeline for churn prediction data.
    
    Args:
        train_df: Training dataframe
        test_df: Testing dataframe
        
    Returns:
        Tuple of (X_train_processed, X_test_processed, y_train, y_test)
    """
    print("🔄 Starting data preprocessing...")
    
    # 1. Encode binary features
    binary_cols = ['International plan', 'Voice mail plan', 'Churn']
    train_encoded = encode_binary_features(train_df, binary_cols)
    test_encoded = encode_binary_features(test_df, binary_cols)
    
    # 2. Separate features and target
    y_train = train_encoded['Churn']
    y_test = test_encoded['Churn']
    X_train = train_encoded.drop('Churn', axis=1)
    X_test = test_encoded.drop('Churn', axis=1)
    
    # 3. Define column types
    categorical_cols = ['State', 'Area code']
    numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    numerical_cols = [col for col in numerical_cols 
                     if col not in categorical_cols + ['International plan', 'Voice mail plan']]
    remainder_cols = ['International plan', 'Voice mail plan']
    
    # 4. Create and apply preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numerical_cols, categorical_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 5. Convert back to DataFrames with proper column names
    feature_names = get_feature_names(preprocessor, numerical_cols, categorical_cols, remainder_cols)
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    
    # 6. Remove highly correlated features
    X_train_final, removed_features = remove_correlated_features(X_train_df)
    X_test_final = X_test_df.drop(columns=removed_features)
    
    print(f"✅ Preprocessing complete!")
    print(f"   Final feature count: {X_train_final.shape[1]}")
    print(f"   Training samples: {X_train_final.shape[0]}")
    print(f"   Testing samples: {X_test_final.shape[0]}")
    
    return X_train_final, X_test_final, y_train, y_test


if __name__ == "__main__":
    # Example usage
    train_path = "../data/churn-bigml-80.csv"
    test_path = "../data/churn-bigml-20.csv"
    
    # Load and preprocess data
    train_df, test_df = load_data(train_path, test_path)
    X_train, X_test, y_train, y_test = preprocess_data(train_df, test_df)
    
    print(f"\n📊 Preprocessing Summary:")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    print(f"   Class distribution: {y_train.value_counts().to_dict()}")