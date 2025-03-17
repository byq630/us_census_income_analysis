from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, average_precision_score, roc_auc_score
import numpy as np

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluates a classification model's performance.

    Parameters:
    - model: estimator
    - X_test: input features
    - y_test: True target
    - threshold: Decision threshold for converting probabilities to class labels (default = 0.5)

    Returns:
    - Dictionary containing accuracy, recall, precision, f1-score, 
      average precision score, and ROC-AUC score.
    """
    y_pred_proba = model.predict_proba(X_test)[:,1]
    y_pred = (np.array(y_pred_proba) >= threshold).astype(int)  # Convert probabilities to binary labels
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Average Precision Score": average_precision_score(y_test, y_pred_proba),
        "ROC-AUC Score": roc_auc_score(y_test, y_pred_proba),
    }
    
    return metrics
