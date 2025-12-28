
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, precision_recall_curve, auc
import joblib

def pr_auc_score(y_true, y_probs):
    """Computes Area Under the Precision-Recall Curve."""
    if y_probs.ndim > 1:
        y_probs = y_probs[:, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)

def train_baseline_model(X_train, y_train):
    """
    Trains a Logistic Regression model with class_weight='balanced'.
    Business Justification: Logistic Regression provides a clear baseline and 
    interpretability. Handling imbalance via class weights ensures the model 
    doesn't ignore the minority (fraud) class.
    """
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_ensemble_model(X_train, y_train):
    """
    Trains a Random Forest model with class_weight='balanced'.
    Ensemble methods like Random Forest are robust to noise and capture 
    non-linear relationships better than linear models.
    """
    model = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42,
        n_jobs=None
    )
    model.fit(X_train, y_train)
    return model

def stratified_cross_validation(model, X, y, k=5):
    """
    Performs Stratified K-Fold cross-validation.
    Metrics: AUC-PR is chosen because it's more informative than ROC-AUC 
    for highly imbalanced datasets.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    scoring = {
        'f1': make_scorer(f1_score),
        'auc_pr': make_scorer(pr_auc_score, response_method='predict_proba')
    }
    
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    
    return {
        'f1_mean': np.mean(cv_results['test_f1']),
        'f1_std': np.std(cv_results['test_f1']),
        'auc_pr_mean': np.mean(cv_results['test_auc_pr']),
        'auc_pr_std': np.std(cv_results['test_auc_pr'])
    }

def save_model(model, path):
    """Saves the trained model to the specified path."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")
