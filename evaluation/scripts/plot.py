"""
Plotting script cho evaluation results
Gộp: plot_roc.py, plot_confusion_matrix.py, plot_loss_curve.py
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
import logging

# Add paths to sys.path
base_dir = Path(__file__).parent.parent.parent  # code/
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "preprocessing"))

from evaluation.core import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_loss_curve,
    get_roc_curve
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_roc_from_evaluation(evaluation_file: str, model_name: str, save_path: Path):
    """
    Plot ROC curve từ evaluation results JSON
    
    Args:
        evaluation_file: Path to evaluation JSON file
        model_name: Model name for title
        save_path: Path to save plot
    """
    with open(evaluation_file, 'r') as f:
        results = json.load(f)
    
    if 'roc_auc' not in results:
        logger.warning("ROC curve requires labels. Please run evaluation with labels.")
        return
    
    # Cần y_true và y_scores để vẽ ROC curve
    # Evaluation JSON không lưu y_true và y_scores, chỉ lưu metrics
    # Cần tính lại từ errors/scores nếu có
    if 'errors' in results and 'labels' in results:
        y_true = np.array(results['labels'])
        y_scores = np.array(results['errors'])
        fpr, tpr, roc_auc = get_roc_curve(y_true, y_scores)
        plot_roc_curve(fpr, tpr, roc_auc, model_name, save_path)
    else:
        logger.warning("Cannot plot ROC curve: missing labels or scores in evaluation results")


def plot_cm_from_evaluation(evaluation_file: str, model_name: str, save_path: Path):
    """
    Plot confusion matrix từ evaluation results JSON
    
    Args:
        evaluation_file: Path to evaluation JSON file
        model_name: Model name for title
        save_path: Path to save plot
    """
    with open(evaluation_file, 'r') as f:
        results = json.load(f)
    
    if 'confusion_matrix' not in results:
        logger.warning("Confusion matrix not found in evaluation results.")
        logger.info("Please run evaluation with ground truth labels.")
        return
    
    # Reconstruct y_true and y_pred từ confusion matrix
    cm_data = results['confusion_matrix']
    tp = cm_data['tp']
    fp = cm_data['fp']
    tn = cm_data['tn']
    fn = cm_data['fn']
    
    # Create dummy arrays for plotting (chỉ để visualize, không chính xác 100%)
    total = tp + fp + tn + fn
    y_true = np.array([0] * tn + [0] * fp + [1] * fn + [1] * tp)
    y_pred = np.array([0] * tn + [1] * fp + [0] * fn + [1] * tp)
    
    metrics = {
        'precision': results.get('precision', 0),
        'recall': results.get('recall', 0),
        'f1_score': results.get('f1_score', 0),
        'accuracy': results.get('accuracy', 0)
    }
    
    plot_confusion_matrix(y_true, y_pred, model_name, save_path, metrics)


def plot_loss_from_history(history_file: str, model_name: str, save_path: Path):
    """
    Plot loss curves từ training history JSON
    
    Args:
        history_file: Path to training history JSON file
        model_name: Model name for title
        save_path: Path to save plot
    """
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    history = data['history']
    epochs = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    plot_loss_curve(epochs, train_loss, val_loss, model_name, save_path)


def plot_from_evaluation_files(model_type: str, dataset_name: str,
                               evaluation_dir: str = "output/evaluation",
                               checkpoint_dir: str = "../training/output/checkpoints",
                               figures_dir: str = "../../figures/chapter_02"):
    """
    Plot tất cả từ evaluation files
    
    Args:
        model_type: "autoencoder" hoặc "logbert"
        dataset_name: "HDFS" hoặc "BGL"
        evaluation_dir: Directory chứa evaluation results
        checkpoint_dir: Directory chứa checkpoints (cho loss curves)
        figures_dir: Directory để lưu figures
    """
    evaluation_file = Path(evaluation_dir) / f"{model_type}_evaluation.json"
    
    if not evaluation_file.exists():
        logger.warning(f"Evaluation file not found: {evaluation_file}")
        return
    
    model_name = f"{model_type.upper()} ({dataset_name})"
    figures_dir = Path(figures_dir)
    
    # Plot ROC curve
    roc_path = figures_dir / f"roc_curve_{model_type}_{dataset_name.lower()}.png"
    plot_roc_from_evaluation(str(evaluation_file), model_name, roc_path)
    
    # Plot confusion matrix
    cm_path = figures_dir / f"confusion_matrix_{model_type}_{dataset_name.lower()}.png"
    plot_cm_from_evaluation(str(evaluation_file), model_name, cm_path)
    
    # Plot loss curve từ checkpoint
    checkpoint_path = Path(checkpoint_dir) / f"{model_type}_{dataset_name.lower()}_local" / "training_history.json"
    if checkpoint_path.exists():
        loss_path = figures_dir / f"{model_type}_loss_curve_{dataset_name.lower()}.png"
        plot_loss_from_history(str(checkpoint_path), model_name, loss_path)
    else:
        logger.warning(f"Training history not found: {checkpoint_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot evaluation results')
    parser.add_argument('--plot_type', type=str, default='all', 
                       choices=['roc', 'cm', 'loss', 'all'],
                       help='Type of plot: roc, cm, loss, or all')
    parser.add_argument('--model_type', type=str, default='both',
                       choices=['autoencoder', 'logbert', 'both'],
                       help='Model type: autoencoder, logbert, or both')
    parser.add_argument('--dataset', type=str, default='both',
                       choices=['HDFS', 'BGL', 'both'],
                       help='Dataset: HDFS, BGL, or both')
    parser.add_argument('--evaluation_dir', type=str, default='output/evaluation',
                       help='Directory containing evaluation results')
    parser.add_argument('--checkpoint_dir', type=str, default='../training/output/checkpoints',
                       help='Directory containing checkpoints')
    parser.add_argument('--figures_dir', type=str, default='../../figures/chapter_02',
                       help='Directory to save figures')
    
    args = parser.parse_args()
    
    # Determine models and datasets to plot
    models = ['autoencoder', 'logbert'] if args.model_type == 'both' else [args.model_type]
    datasets = ['HDFS', 'BGL'] if args.dataset == 'both' else [args.dataset]
    
    for model_type in models:
        for dataset_name in datasets:
            logger.info(f"\nPlotting {model_type} - {dataset_name}...")
            plot_from_evaluation_files(
                model_type=model_type,
                dataset_name=dataset_name,
                evaluation_dir=args.evaluation_dir,
                checkpoint_dir=args.checkpoint_dir,
                figures_dir=args.figures_dir
            )
    
    logger.info("\n✅ Plotting completed!")


if __name__ == "__main__":
    main()

