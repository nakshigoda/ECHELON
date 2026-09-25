"""
Main training pipeline for the ECHELON predictive maintenance project.
"""

import os
import numpy as np
import xgboost as xgb
import joblib

from src.preprocessing import load_and_inspect_data, preprocess_data, split_data, scale_features
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from src.prediction import predict

def main():
    # Create models/ directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print('=' * 60)
    print('Step 1: Load and inspect data')
    print('=' * 60)
    df = load_and_inspect_data('data/ai4i2020.csv')
    
    print('\n' + '=' * 60)
    print('Step 2: Preprocess data')
    print('=' * 60)
    X, y = preprocess_data(df)
    feature_names = list(X.columns)
    print(f'Feature columns: {feature_names}')
    print(f'Target: {y.name}')
    
    print('\n' + '=' * 60)
    print('Step 3: Split data')
    print('=' * 60)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')
    print(f'Class distribution in train:\n{y_train.value_counts()}')
    print(f'Class distribution in test:\n{y_test.value_counts()}')
    
    print('\n' + '=' * 60)
    print('Step 4: Scale features')
    print('=' * 60)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test, save_path='models/')
    
    print('\n' + '=' * 60)
    print('Step 5: Train XGBoost Classifier')
    print('=' * 60)
    # Calculate scale_pos_weight to handle class imbalance
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count
    print(f'Calculated scale_pos_weight: {scale_pos_weight:.2f}')
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train_scaled, y_train)
    model_path = 'models/xgboost_model.joblib'
    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')
    
    print('\n' + '=' * 60)
    print('Step 6: Evaluate model')
    print('=' * 60)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    metrics = evaluate_model(y_test, y_pred, y_proba)
    plot_confusion_matrix(y_test, y_pred, save_path='models/confusion_matrix.png')
    plot_roc_curve(y_test, y_proba, save_path='models/roc_curve.png')
    plot_feature_importance(model, feature_names, save_path='models/feature_importance.png')
    
    print('\n' + '=' * 60)
    print('Step 7: Sample prediction')
    print('=' * 60)
    sample_input = {
        'Type': 'M', 
        'Air temperature [K]': 298.1, 
        'Process temperature [K]': 308.6, 
        'Rotational speed [rpm]': 1551, 
        'Torque [Nm]': 42.8, 
        'Tool wear [min]': 0
    }
    prediction = predict(sample_input)
    print(f'Prediction result: {prediction}')
    
    print('\n' + '=' * 60)
    print('Step 8: Print summary')
    print('=' * 60)
    print('Files saved to models/:')
    for file in os.listdir('models'):
        print(f'- {file}')
    print('\nTraining pipeline completed successfully!')

if __name__ == '__main__':
    main()
