
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, auc, f1_score
)

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluates the model and prints key metrics.
    """
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # F1 Score
    f1 = f1_score(y_test, y_pred)
    
    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    auc_pr = auc(recall, precision)
    
    print(f"--- {model_name} Evaluation ---")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return {
        'f1': f1,
        'auc_pr': auc_pr,
        'cm': cm
    }

def compare_models(results_dict):
    """
    Compares metrics of multiple models side-by-side.
    results_dict: {model_name: {'f1': val, 'auc_pr': val}}
    """
    df_results = pd.DataFrame(results_dict).T
    
    df_results[['f1', 'auc_pr']].plot(kind='bar', figsize=(10, 6))
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
    return df_results
