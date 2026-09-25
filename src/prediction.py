import pandas as pd
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(models_path='models/'):
    """
    Loads the trained XGBoost model and associated artifacts.

    Args:
        models_path (str): The path to the directory containing the model artifacts.

    Returns:
        dict: A dictionary containing the 'model', 'scaler', 'label_encoder', and 'feature_names'.
              Returns None if an error occurs.
    """
    try:
        model_path = os.path.join(models_path, 'xgboost_model.joblib')
        scaler_path = os.path.join(models_path, 'scaler.joblib')
        le_path = os.path.join(models_path, 'label_encoder.joblib')
        features_path = os.path.join(models_path, 'feature_names.joblib')

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(le_path)
        feature_names = joblib.load(features_path)

        return {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_names': feature_names
        }
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        return None

def predict(input_data, model_artifacts=None, models_path='models/'):
    """
    Makes a prediction for a single data instance.

    Args:
        input_data (dict): A dictionary containing the input features.
            Keys required: 'Type', 'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
        model_artifacts (dict, optional): A dictionary of pre-loaded model artifacts.
            If None, the function will attempt to load them using load_model().
        models_path (str, optional): Path to model directory if artifacts need to be loaded.

    Returns:
        dict: A dictionary containing 'prediction' (0 or 1), 'failure_probability' (float),
              'no_failure_probability' (float), and 'status' (string).
              Returns None if an error occurs.
    """
    try:
        if model_artifacts is None:
            model_artifacts = load_model(models_path)
            if model_artifacts is None:
                raise ValueError("Could not load model artifacts.")

        model = model_artifacts['model']
        scaler = model_artifacts['scaler']
        label_encoder = model_artifacts['label_encoder']
        feature_names = model_artifacts['feature_names']

        # Convert to DataFrame
        df = pd.DataFrame([input_data])
        
        # Sanitize column names to match training preprocessing
        # (XGBoost does not allow [ ] < in feature names)
        df.columns = df.columns.str.replace('[', '(', regex=False).str.replace(']', ')', regex=False)
        if 'Type' in df.columns:
            # Handle unknown labels gracefully if needed, but here we assume 'L', 'M', 'H'
            df['Type'] = label_encoder.transform(df['Type'])
        else:
             raise ValueError("Input data must contain 'Type'.")

        # Ensure correct column order
        missing_cols = set(feature_names) - set(df.columns)
        if missing_cols:
             raise ValueError(f"Input data is missing columns: {missing_cols}")

        df = df[feature_names]

        # Scale features
        X_scaled = scaler.transform(df)

        # Predict
        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0]

        failure_prob = float(probabilities[1])
        no_failure_prob = float(probabilities[0])

        status = 'Failure Predicted' if prediction == 1 else 'No Failure Predicted'

        return {
            'prediction': prediction,
            'failure_probability': failure_prob,
            'no_failure_probability': no_failure_prob,
            'status': status
        }

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None
