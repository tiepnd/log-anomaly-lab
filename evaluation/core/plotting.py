"""
Plotting utilities for evaluation
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float, 
                   model_name: str, save_path: Path):
    """
    Plot ROC curve
    
    Args:
        fpr: False Positive Rate
        tpr: True Positive Rate
        roc_auc: ROC-AUC score
        model_name: Model name for title
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved ROC curve to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str, save_path: Path,
                         metrics: Optional[Dict] = None):
    """
    Plot confusion matrix
    
    Args:
        y_true: Ground truth labels
        y_pred: Predictions
        model_name: Model name for title
        save_path: Path to save plot
        metrics: Optional metrics dict (precision, recall, f1_score, accuracy)
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    cm_array = np.array([[tn, fp],
                         [fn, tp]])
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'],
                cbar_kws={'label': 'Count'}, 
                annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.xlabel('Predicted', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=14, fontweight='bold')
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    
    # Add metrics text
    if metrics:
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1 = metrics.get('f1_score', 0)
        accuracy = metrics.get('accuracy', 0)
        
        metrics_text = f'Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}\nAccuracy: {accuracy:.4f}'
        plt.text(1.5, 0.5, metrics_text, 
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='center')
    
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved confusion matrix to {save_path}")
    plt.close()


def plot_loss_curve(epochs: list, train_loss: list, val_loss: list,
                   model_name: str, save_path: Path):
    """
    Plot loss curves từ training history
    
    Args:
        epochs: List of epoch numbers
        train_loss: List of training losses
        val_loss: List of validation losses
        model_name: Model name for title
        save_path: Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot loss curves
    plt.plot(epochs, train_loss, label='Train Loss', marker='o', 
            markersize=4, linewidth=2, color='#2E86AB')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='s', 
            markersize=4, linewidth=2, color='#A23B72')
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title(f'Training and Validation Loss - {model_name}', 
             fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add best epoch annotation
    best_epoch_idx = np.argmin(val_loss)
    best_val_loss = val_loss[best_epoch_idx]
    best_epoch = epochs[best_epoch_idx]
    
    plt.axvline(best_epoch, color='red', linestyle='--', alpha=0.5, 
               label=f'Best Epoch: {best_epoch}')
    plt.plot(best_epoch, best_val_loss, 'ro', markersize=10, 
            label=f'Best Val Loss: {best_val_loss:.6f}')
    
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved loss curve to {save_path}")
    plt.close()


def plot_score_distribution(scores: np.ndarray, threshold: float,
                           model_name: str, save_path: Path,
                           label: str = "Score"):
    """
    Plot score/error distribution với threshold line
    
    Args:
        scores: Array of scores/errors
        threshold: Threshold value
        model_name: Model name for title
        save_path: Path to save plot
        label: Label for x-axis
    """
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, edgecolor='black', alpha=0.7, label='Distribution')
    plt.axvline(threshold, color='r', linestyle='--', linewidth=2, 
               label=f'Threshold: {threshold:.6f}')
    plt.xlabel(label, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{model_name} - Score/Error Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved distribution plot to {save_path}")
    plt.close()

