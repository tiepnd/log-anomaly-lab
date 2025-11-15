"""
Metrics calculation utilities
"""
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve, auc
)
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predictions
        y_scores: Prediction scores (for ROC-AUC, PR-AUC)
    
    Returns:
        Dict với metrics
    """
    # Binary metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    results = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        'confusion_matrix': {
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        },
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    # ROC-AUC và PR-AUC nếu có scores
    if y_scores is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
            results['roc_auc'] = float(roc_auc)
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            results['roc_auc'] = 0.0
        
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall_curve, precision_curve)
            results['pr_auc'] = float(pr_auc)
        except Exception as e:
            logger.warning(f"Could not calculate PR-AUC: {e}")
            results['pr_auc'] = 0.0
    
    return results


def get_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate ROC curve
    
    Args:
        y_true: Ground truth labels
        y_scores: Prediction scores
    
    Returns:
        Tuple (fpr, tpr, roc_auc)
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    
    return fpr, tpr, roc_auc


def get_pr_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate Precision-Recall curve
    
    Args:
        y_true: Ground truth labels
        y_scores: Prediction scores
    
    Returns:
        Tuple (precision, recall, pr_auc)
    """
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall_curve, precision_curve)
    
    return precision_curve, recall_curve, pr_auc

