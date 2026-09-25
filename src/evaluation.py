import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend for non-interactive environments - must be set before pyplot import
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

def evaluate_model(y_true, y_pred, y_proba):
    """
    Evaluate a binary classification model for predicting machine failure.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_proba (array-like): Predicted probabilities for the positive class (Machine failure = 1).

    Returns:
        dict: A dictionary containing accuracy, precision, recall, f1_score, and roc_auc.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba)
    }

    print("Model Evaluation Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize()}: {value:.4f}")

    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix using seaborn heatmap.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_roc_curve(y_true, y_proba, save_path=None):
    """
    Plot ROC curve with AUC score in legend.

    Args:
        y_true (array-like): True labels.
        y_proba (array-like): Predicted probabilities for the positive class.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
    plt.close()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot feature importance as a horizontal bar chart.

    Args:
        model: Trained model with feature_importances_ attribute (e.g., XGBoost, RandomForest).
        feature_names (list): List of feature names.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    importances = model.feature_importances_
    
    # Create a DataFrame for easy sorting and plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
    plt.close()
