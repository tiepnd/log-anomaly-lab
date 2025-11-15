"""
Evaluation script cho Autoencoder và LogBERT
Tính các metrics: Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
"""

import os
import sys
import json
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve, auc
)

# Add paths to sys.path
base_dir = Path(__file__).parent.parent.parent  # code/
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "preprocessing"))

from models.autoencoder import create_autoencoder
from models.logbert import create_logbert
from transformers import AutoTokenizer
from preprocessing.core.parser import LogParser
from preprocessing.core.pipeline import LogPreprocessingPipeline

# Import core evaluation modules
from evaluation.core import (
    load_autoencoder_model,
    load_logbert_model,
    load_ground_truth_labels,
    map_logs_to_labels,
    calculate_metrics,
    get_roc_curve,
    plot_roc_curve,
    plot_confusion_matrix,
    load_threshold
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Functions đã được move vào core modules, import từ đó


def evaluate_autoencoder(model_path: str,
                        test_embeddings: np.ndarray,
                        threshold: float,
                        labels: Optional[np.ndarray] = None,
                        device: str = "cpu",
                        save_dir: str = "evaluation",
                        plot_roc: bool = True,
                        plot_cm: bool = True) -> Dict:
    """
    Evaluate Autoencoder model
    
    Args:
        model_path: Path to model checkpoint
        test_embeddings: Test embeddings
        threshold: Threshold for anomaly detection
        labels: Ground truth labels (optional)
        device: Device
        save_dir: Directory to save results
    
    Returns:
        Evaluation results dict
    """
    logger.info("\n" + "="*70)
    logger.info("EVALUATION - AUTOENCODER")
    logger.info("="*70)
    
    # Load model
    model, config = load_model_autoencoder(model_path, device)
    
    # Calculate reconstruction errors
    logger.info("Calculating reconstruction errors on test set...")
    errors = []
    batch_size = 64
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_embeddings), batch_size), desc="Computing errors"):
            batch = torch.FloatTensor(test_embeddings[i:i+batch_size]).to(device)
            error = model.get_reconstruction_error(batch)
            errors.extend(error.cpu().numpy().tolist())
    
    errors = np.array(errors)
    
    # Predictions
    predictions = (errors > threshold).astype(int)
    
    results = {
        'threshold': float(threshold),
        'predictions': predictions.tolist(),
        'errors': errors.tolist(),
        'error_statistics': {
            'mean': float(errors.mean()),
            'std': float(errors.std()),
            'min': float(errors.min()),
            'max': float(errors.max())
        },
        'prediction_statistics': {
            'anomaly_count': int(np.sum(predictions)),
            'normal_count': int(np.sum(1 - predictions)),
            'anomaly_rate': float(np.mean(predictions))
        }
    }
    
    # Calculate metrics nếu có labels
    if labels is not None and len(labels) > 0:
        logger.info("Calculating metrics with ground truth labels...")
        
        # Use core metrics module
        metrics = calculate_metrics(labels, predictions, errors)
        results.update(metrics)
        
        logger.info(f"\n📊 Evaluation Metrics:")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        logger.info(f"   F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
        logger.info(f"   PR-AUC: {metrics.get('pr_auc', 0):.4f}")
        logger.info(f"\n   Confusion Matrix:")
        cm = metrics['confusion_matrix']
        logger.info(f"   TP: {cm['tp']}, FP: {cm['fp']}")
        logger.info(f"   TN: {cm['tn']}, FN: {cm['fn']}")
    else:
        logger.info("No ground truth labels provided. Skipping metrics calculation.")
    
    # Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = save_dir / "autoencoder_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Evaluation results saved to {results_path}")
    
    # Plot ROC curve và Confusion Matrix nếu có labels
    if labels is not None and len(labels) > 0:
        if plot_roc:
            roc_path = Path(save_dir) / f"roc_curve_autoencoder.png"
            fpr, tpr, roc_auc = get_roc_curve(labels, errors)
            plot_roc_curve(fpr, tpr, roc_auc, "Autoencoder", roc_path)
        
        if plot_cm:
            cm_path = Path(save_dir) / f"confusion_matrix_autoencoder.png"
            metrics_dict = results if 'precision' in results else None
            plot_confusion_matrix(labels, predictions, "Autoencoder", cm_path, metrics_dict)
    
    return results


def evaluate_logbert(model_path: str,
                     test_templates: List[str],
                     tokenizer,
                     threshold: float,
                     labels: Optional[np.ndarray] = None,
                     device: str = "cpu",
                     max_length: int = 128,
                     save_dir: str = "evaluation",
                     plot_roc: bool = True,
                     plot_cm: bool = True) -> Dict:
    """
    Evaluate LogBERT model
    
    Args:
        model_path: Path to model checkpoint
        test_templates: Test templates
        tokenizer: BERT tokenizer
        threshold: Threshold for anomaly detection
        labels: Ground truth labels (optional)
        device: Device
        max_length: Max sequence length
        save_dir: Directory to save results
    
    Returns:
        Evaluation results dict
    """
    logger.info("\n" + "="*70)
    logger.info("EVALUATION - LOGBERT")
    logger.info("="*70)
    
    # Load model
    model, config = load_model_logbert(model_path, device)
    
    # Calculate anomaly scores
    logger.info("Calculating anomaly scores on test set...")
    scores = []
    batch_size = 16
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_templates), batch_size), desc="Computing scores"):
            batch_templates = test_templates[i:i+batch_size]
            
            encoded = tokenizer(
                batch_templates,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            batch_scores = model.get_anomaly_score(input_ids, attention_mask)
            scores.extend(batch_scores.cpu().numpy().tolist())
    
    scores = np.array(scores)
    
    # Predictions
    predictions = (scores > threshold).astype(int)
    
    results = {
        'threshold': float(threshold),
        'predictions': predictions.tolist(),
        'scores': scores.tolist(),
        'score_statistics': {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max())
        },
        'prediction_statistics': {
            'anomaly_count': int(np.sum(predictions)),
            'normal_count': int(np.sum(1 - predictions)),
            'anomaly_rate': float(np.mean(predictions))
        }
    }
    
    # Calculate metrics nếu có labels
    if labels is not None and len(labels) > 0:
        logger.info("Calculating metrics with ground truth labels...")
        
        # Use core metrics module
        metrics = calculate_metrics(labels, predictions, scores)
        results.update(metrics)
        
        logger.info(f"\n📊 Evaluation Metrics:")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall: {metrics['recall']:.4f}")
        logger.info(f"   F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
        logger.info(f"   PR-AUC: {metrics.get('pr_auc', 0):.4f}")
        logger.info(f"\n   Confusion Matrix:")
        cm = metrics['confusion_matrix']
        logger.info(f"   TP: {cm['tp']}, FP: {cm['fp']}")
        logger.info(f"   TN: {cm['tn']}, FN: {cm['fn']}")
    else:
        logger.info("No ground truth labels provided. Skipping metrics calculation.")
    
    # Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = save_dir / "logbert_evaluation.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Evaluation results saved to {results_path}")
    
    # Plot ROC curve và Confusion Matrix nếu có labels
    if labels is not None and len(labels) > 0:
        if plot_roc:
            roc_path = Path(save_dir) / f"roc_curve_logbert.png"
            fpr, tpr, roc_auc = get_roc_curve(labels, scores)
            plot_roc_curve(fpr, tpr, roc_auc, "LogBERT", roc_path)
        
        if plot_cm:
            cm_path = Path(save_dir) / f"confusion_matrix_logbert.png"
            metrics_dict = results if 'precision' in results else None
            plot_confusion_matrix(labels, predictions, "LogBERT", cm_path, metrics_dict)
    
    return results


# Plotting functions đã được move vào core/plotting.py, import từ đó


def evaluate_local(model_type: str,
                   dataset_name: str,
                   checkpoint_dir: str = "../training/output/checkpoints",
                   threshold_dir: str = "../training/output/thresholds",
                   log_file: str = None,
                   label_file: str = None,
                   bert_model_name: str = "distilbert-base-uncased",
                   device: str = "cpu",
                   save_dir: str = "output/evaluation") -> Dict:
    """
    Evaluate model trên local dataset với ground truth labels
    
    Args:
        model_type: "autoencoder" hoặc "logbert"
        dataset_name: "HDFS" hoặc "BGL"
        checkpoint_dir: Directory chứa checkpoints
        threshold_dir: Directory chứa thresholds
        log_file: Log file (auto-detect nếu None)
        label_file: Path to ground truth labels CSV (auto-detect nếu None)
        bert_model_name: BERT model name (cho LogBERT)
        device: Device
        save_dir: Directory để lưu results
    """
    # Tìm checkpoint
    checkpoint_path = Path(checkpoint_dir) / f"{model_type}_{dataset_name.lower()}_local" / "best_model.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load threshold
    threshold = load_threshold(model_type, dataset_name, threshold_dir)
    
    if threshold is None:
        logger.warning("Threshold not found, will calculate from validation set")
        # Có thể tính threshold từ validation set nếu cần
    
    # Tìm log file
    if log_file is None:
        base_dir = Path(__file__).parent.parent.parent
        log_file = base_dir / "datasets" / f"{dataset_name}_2k.log"
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
    
    log_file = str(log_file)
    
    # Load ground truth labels nếu có
    labels_dict = None
    if label_file is None and dataset_name == "HDFS":
        # Auto-detect label file cho HDFS
        base_dir = Path(__file__).parent.parent.parent
        label_file = base_dir / "datasets" / "HDFS_v1" / "preprocessed" / "anomaly_label.csv"
        if label_file.exists():
            labels_dict = load_ground_truth_labels(str(label_file))
        else:
            logger.warning(f"Label file not found: {label_file}")
    elif label_file:
        labels_dict = load_ground_truth_labels(label_file)
    
    # Load và process logs
    logger.info("Loading test data...")
    persistence_dir = Path(__file__).parent.parent / "output" / "drain3_state" / dataset_name.lower()
    parser = LogParser(persistence_dir=str(persistence_dir))
    parsed_logs = parser.parse_dataset(log_file, dataset_type=dataset_name)
    
    # Map logs với labels nếu có
    label_indices = None
    test_labels = None
    if labels_dict:
        label_indices, test_labels = map_logs_to_labels(parsed_logs, labels_dict)
        logger.info(f"Found labels for {len(label_indices)} log entries")
        logger.info(f"Label distribution: {np.sum(test_labels == 0)} Normal, {np.sum(test_labels == 1)} Anomaly")
    
    if model_type == "autoencoder":
        # Process cho Autoencoder
        pipeline = LogPreprocessingPipeline(
            tokenizer_method="word",
            embedder_method="word2vec",
            embedding_dim=128,
            vocab_size=5000
        )
        pipeline.fit(parsed_logs, dataset_type=dataset_name)
        processed_logs = pipeline.process_parsed_logs(parsed_logs)
        
        # Extract embeddings
        embeddings = []
        embedding_indices = []  # Track original log indices
        for idx, log in enumerate(processed_logs):
            if log.get('embedded') is not None:
                embeddings.append(log['embedded'])
                embedding_indices.append(idx)
        
        embeddings = np.array(embeddings)
        embedding_indices = np.array(embedding_indices)
        
        # Filter embeddings có labels nếu có
        if label_indices is not None and test_labels is not None:
            # Find intersection: embeddings có labels
            mask = np.isin(embedding_indices, label_indices)
            test_embeddings = embeddings[mask]
            test_embedding_indices = embedding_indices[mask]
            
            # Map labels với filtered embeddings
            label_map = {idx: label for idx, label in zip(label_indices, test_labels)}
            test_labels_filtered = np.array([label_map[idx] for idx in test_embedding_indices])
            
            logger.info(f"Test embeddings with labels: {len(test_embeddings)}")
            logger.info(f"Label distribution: {np.sum(test_labels_filtered == 0)} Normal, {np.sum(test_labels_filtered == 1)} Anomaly")
        else:
            # Không có labels, lấy 50% cuối làm test set
            n_test = int(len(embeddings) * 0.5)
            test_embeddings = embeddings[-n_test:]
            test_labels_filtered = None
            logger.info(f"Test embeddings (no labels): {len(test_embeddings)}")

        # Evaluate
        results = evaluate_autoencoder(
            str(checkpoint_path),
            test_embeddings,
            threshold,
            labels=test_labels_filtered,  # Use ground truth labels if available
            device=device,
            save_dir=save_dir
        )
        
        # Plot ROC curve nếu có labels
        # if labels is not None...
        
    else:
        # Process cho LogBERT
        templates = []
        template_indices = []  # Track original log indices
        for idx, log in enumerate(parsed_logs):
            if log.get('template'):
                templates.append(log['template'])
                template_indices.append(idx)
        
        template_indices = np.array(template_indices)
        
        # Filter templates có labels nếu có
        if label_indices is not None and test_labels is not None:
            # Find intersection: templates có labels
            mask = np.isin(template_indices, label_indices)
            test_templates = [templates[i] for i in range(len(templates)) if mask[i]]
            test_template_indices = template_indices[mask]
            
            # Map labels với filtered templates
            label_map = {idx: label for idx, label in zip(label_indices, test_labels)}
            test_labels_filtered = np.array([label_map[idx] for idx in test_template_indices])
            
            logger.info(f"Test templates with labels: {len(test_templates)}")
            logger.info(f"Label distribution: {np.sum(test_labels_filtered == 0)} Normal, {np.sum(test_labels_filtered == 1)} Anomaly")
        else:
            # Không có labels, lấy 50% cuối làm test set
            n_test = int(len(templates) * 0.5)
            test_templates = templates[-n_test:]
            test_labels_filtered = None
            logger.info(f"Test templates (no labels): {len(test_templates)}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        
        # Evaluate
        results = evaluate_logbert(
            str(checkpoint_path),
            test_templates,
            tokenizer,
            threshold,
            labels=test_labels_filtered,  # Use ground truth labels if available
            device=device,
            save_dir=save_dir
        )
    
    return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate models')
    parser.add_argument('--model_type', type=str, required=True, choices=['autoencoder', 'logbert'],
                       help='Model type: autoencoder or logbert')
    parser.add_argument('--dataset', type=str, default='HDFS', choices=['HDFS', 'BGL'],
                       help='Dataset name')
    parser.add_argument('--checkpoint_dir', type=str, default='../training/output/checkpoints',
                       help='Directory containing checkpoints')
    parser.add_argument('--threshold_dir', type=str, default='../training/output/thresholds',
                       help='Directory containing thresholds')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to log file (auto-detect if None)')
    parser.add_argument('--label_file', type=str, default=None,
                       help='Path to ground truth labels CSV (auto-detect for HDFS if None)')
    parser.add_argument('--bert_model', type=str, default='distilbert-base-uncased',
                       help='BERT model name (for LogBERT)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: cpu or cuda')
    parser.add_argument('--save_dir', type=str, default='output/evaluation',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Evaluate
    results = evaluate_local(
        model_type=args.model_type,
        dataset_name=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        threshold_dir=args.threshold_dir,
        log_file=args.log_file,
        label_file=args.label_file,
        bert_model_name=args.bert_model,
        device=args.device,
        save_dir=args.save_dir
    )
    
    logger.info("\n✅ Evaluation completed!")
    
    # Print summary
    if 'f1_score' in results:
        logger.info("\n📊 Summary:")
        logger.info(f"   Precision: {results['precision']:.4f}")
        logger.info(f"   Recall: {results['recall']:.4f}")
        logger.info(f"   F1-Score: {results['f1_score']:.4f}")
        logger.info(f"   Accuracy: {results['accuracy']:.4f}")
        logger.info(f"   ROC-AUC: {results['roc_auc']:.4f}")


if __name__ == "__main__":
    main()

