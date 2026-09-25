import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def load_and_inspect_data(filepath):
    """
    Load the dataset and print inspection information.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(filepath)
    print("--- Data Inspection ---")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicates:")
    print(df.duplicated().sum())
    
    print("\nClass Distribution (Machine failure):")
    if 'Machine failure' in df.columns:
        print(df['Machine failure'].value_counts())
    else:
        print("Target column 'Machine failure' not found.")
        
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    return df

def preprocess_data(df):
    """
    Preprocess the data: drop unnecessary columns, encode categorical features,
    and separate features and target.
    
    Args:
        df (pd.DataFrame): Raw dataset.
        
    Returns:
        tuple: Features (X) and target (y).
    """
    # Drop unnecessary columns
    cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df_processed = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Sanitize column names: XGBoost does not allow [ ] < in feature names
    df_processed.columns = df_processed.columns.str.replace('[', '(', regex=False).str.replace(']', ')', regex=False)
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Encode 'Type' column (product quality variant: L=Low, M=Medium, H=High)
    if 'Type' in df_processed.columns:
        le = LabelEncoder()
        # H=0, L=1, M=2 matches the alphabetical order of LabelEncoder
        df_processed['Type'] = le.fit_transform(df_processed['Type'])
        # Save LabelEncoder artifact
        joblib.dump(le, 'models/label_encoder.joblib')

    # Select features and target
    feature_cols = ['Type', 'Air temperature (K)', 'Process temperature (K)', 
                    'Rotational speed (rpm)', 'Torque (Nm)', 'Tool wear (min)']
    
    # Save feature names
    joblib.dump(feature_cols, 'models/feature_names.joblib')
    
    X = df_processed[feature_cols]
    y = df_processed['Machine failure']
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Perform a stratified train/test split.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed.
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def scale_features(X_train, X_test, save_path='models/'):
    """
    Scale numeric features using StandardScaler and save the scaler artifact.
    
    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        save_path (str): Directory to save the scaler artifact.
        
    Returns:
        tuple: Scaled X_train, scaled X_test as pandas DataFrames.
    """
    os.makedirs(save_path, exist_ok=True)
    
    scaler = StandardScaler()
    
    # Fit and transform training data, transform test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names and index
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Save the scaler artifact
    joblib.dump(scaler, os.path.join(save_path, 'scaler.joblib'))
    
    return X_train_scaled, X_test_scaled

def preprocess_for_prediction(input_data, models_path='models/'):
    """
    Load preprocessing artifacts and transform new data for prediction.
    Must be exactly consistent with training preprocessing.
    
    Args:
        input_data (pd.DataFrame): New data to predict.
        models_path (str): Directory where artifacts are saved.
        
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction.
    """
    # Load artifacts
    le = joblib.load(os.path.join(models_path, 'label_encoder.joblib'))
    scaler = joblib.load(os.path.join(models_path, 'scaler.joblib'))
    feature_names = joblib.load(os.path.join(models_path, 'feature_names.joblib'))
    
    # Ensure all required features are present
    df = input_data.copy()
    
    # Sanitize column names to match training preprocessing
    df.columns = df.columns.str.replace('[', '(', regex=False).str.replace(']', ')', regex=False)
    
    # Encode 'Type' if present
    if 'Type' in df.columns:
        # Depending on input_data, Type might already be encoded or not, 
        # assume raw input strings (L, M, H)
        df['Type'] = le.transform(df['Type'])
        
    # Select features
    X = df[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=X.index)
    
    return X_scaled
