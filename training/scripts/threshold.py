"""
Threshold Selection cho Anomaly Detection
Test nhiều phương pháp và chọn threshold tốt nhất trên validation set
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc

# Add paths to sys.path
base_dir = Path(__file__).parent.parent.parent  # code/
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "preprocessing"))


from models.autoencoder import create_autoencoder
from models.logbert import create_logbert
from transformers import AutoTokenizer

# Import preprocessing modules
from preprocessing.core.parser import LogParser
from preprocessing.core.pipeline import LogPreprocessingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_autoencoder_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained Autoencoder model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = create_autoencoder(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded Autoencoder model from {checkpoint_path}")
    return model, config


def load_logbert_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained LogBERT model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = create_logbert(
        bert_model_name=config['bert_model_name'],
        task=config['task'],
        num_labels=2,
        use_pretrained=False  # Will load from checkpoint
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded LogBERT model from {checkpoint_path}")
    return model, config


def calculate_reconstruction_errors(model, embeddings, device, batch_size=64):
    """Tính reconstruction errors cho Autoencoder"""
    errors = []
    
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.FloatTensor(embeddings[i:i+batch_size]).to(device)
            error = model.get_reconstruction_error(batch)
            errors.extend(error.cpu().numpy().tolist())
    
    return np.array(errors)


def calculate_anomaly_scores(model, templates, tokenizer, device, max_length=128, batch_size=16):
    """Tính anomaly scores cho LogBERT"""
    scores = []
    
    with torch.no_grad():
        for i in range(0, len(templates), batch_size):
            batch_templates = templates[i:i+batch_size]
            
            # Tokenize
            encoded = tokenizer(
                batch_templates,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Get anomaly scores
            batch_scores = model.get_anomaly_score(input_ids, attention_mask)
            scores.extend(batch_scores.cpu().numpy().tolist())
    
    return np.array(scores)


def percentile_threshold(scores: np.ndarray, percentile: float = 95.0) -> float:
    """Threshold dựa trên percentile"""
    return np.percentile(scores, percentile)


def std_threshold(scores: np.ndarray, n_std: float = 2.0) -> float:
    """Threshold dựa trên standard deviation"""
    mean = np.mean(scores)
    std = np.std(scores)
    return mean + n_std * std


def iqr_threshold(scores: np.ndarray, multiplier: float = 1.5) -> float:
    """Threshold dựa trên Interquartile Range (IQR)"""
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    return q3 + multiplier * iqr


def optimal_threshold_f1(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Tìm optimal threshold dựa trên F1-score
    Chỉ dùng khi có ground truth labels
    """
    if labels is None or len(labels) == 0:
        return None, 0.0
    
    # Tính precision-recall curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # Tính F1-score cho mỗi threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Tìm threshold với F1 cao nhất
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return optimal_threshold, best_f1


def evaluate_threshold(scores: np.ndarray, threshold: float, labels: Optional[np.ndarray] = None) -> Dict:
    """Đánh giá threshold"""
    predictions = (scores > threshold).astype(int)
    
    results = {
        'threshold': threshold,
        'anomaly_count': int(np.sum(predictions)),
        'normal_count': int(np.sum(1 - predictions)),
        'anomaly_rate': float(np.mean(predictions))
    }
    
    # Nếu có labels, tính metrics
    if labels is not None and len(labels) > 0:
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        results.update({
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        })
    
    return results


def select_threshold_autoencoder(model_path: str,
                                validation_embeddings: np.ndarray,
                                labels: Optional[np.ndarray] = None,
                                device: str = "cpu",
                                save_dir: str = "output/thresholds") -> Dict:
    """
    Select threshold cho Autoencoder
    
    Args:
        model_path: Đường dẫn đến model checkpoint
        validation_embeddings: Validation embeddings
        labels: Ground truth labels (optional, nếu có thì dùng optimal threshold)
        device: Device
        save_dir: Directory để lưu results
    
    Returns:
        Dict với selected threshold và evaluation results
    """
    logger.info("\n" + "="*70)
    logger.info("THRESHOLD SELECTION - AUTOENCODER")
    logger.info("="*70)
    
    # Load model
    model, config = load_autoencoder_model(model_path, device)
    
    # Calculate reconstruction errors
    logger.info("Calculating reconstruction errors on validation set...")
    errors = calculate_reconstruction_errors(model, validation_embeddings, device)
    
    logger.info(f"Reconstruction errors: mean={errors.mean():.6f}, std={errors.std():.6f}")
    logger.info(f"Min={errors.min():.6f}, Max={errors.max():.6f}")
    
    # Test các threshold methods
    thresholds = {}
    evaluations = {}
    
    # 1. Percentile-based thresholds
    logger.info("\nTesting percentile-based thresholds...")
    for percentile in [90, 95, 99, 99.5, 99.9]:
        threshold = percentile_threshold(errors, percentile)
        thresholds[f'percentile_{percentile}'] = threshold
        eval_result = evaluate_threshold(errors, threshold, labels)
        evaluations[f'percentile_{percentile}'] = eval_result
        logger.info(f"  {percentile}th percentile: threshold={threshold:.6f}, "
                   f"anomaly_rate={eval_result['anomaly_rate']:.2%}")
    
    # 2. Standard deviation thresholds
    logger.info("\nTesting standard deviation thresholds...")
    for n_std in [1.5, 2.0, 2.5, 3.0]:
        threshold = std_threshold(errors, n_std)
        thresholds[f'std_{n_std}'] = threshold
        eval_result = evaluate_threshold(errors, threshold, labels)
        evaluations[f'std_{n_std}'] = eval_result
        logger.info(f"  Mean + {n_std}*std: threshold={threshold:.6f}, "
                   f"anomaly_rate={eval_result['anomaly_rate']:.2%}")
    
    # 3. IQR threshold
    logger.info("\nTesting IQR threshold...")
    threshold = iqr_threshold(errors, multiplier=1.5)
    thresholds['iqr_1.5'] = threshold
    eval_result = evaluate_threshold(errors, threshold, labels)
    evaluations['iqr_1.5'] = eval_result
    logger.info(f"  IQR (1.5x): threshold={threshold:.6f}, "
               f"anomaly_rate={eval_result['anomaly_rate']:.2%}")
    
    # 4. Optimal threshold (nếu có labels)
    if labels is not None and len(labels) > 0:
        logger.info("\nFinding optimal threshold based on F1-score...")
        optimal_thresh, best_f1 = optimal_threshold_f1(errors, labels)
        if optimal_thresh is not None:
            thresholds['optimal_f1'] = optimal_thresh
            eval_result = evaluate_threshold(errors, optimal_thresh, labels)
            evaluations['optimal_f1'] = eval_result
            logger.info(f"  Optimal (F1): threshold={optimal_thresh:.6f}, "
                       f"F1={best_f1:.4f}, precision={eval_result['precision']:.4f}, "
                       f"recall={eval_result['recall']:.4f}")
    
    # Chọn threshold tốt nhất
    if labels is not None and len(labels) > 0:
        # Chọn dựa trên F1-score
        best_method = max(evaluations.items(), key=lambda x: x[1].get('f1_score', 0))
    else:
        # Chọn dựa trên percentile 95 (thường dùng)
        best_method = ('percentile_95', evaluations['percentile_95'])
    
    selected_method = best_method[0]
    selected_threshold = thresholds[selected_method]
    selected_eval = best_method[1]
    
    logger.info("\n" + "-"*70)
    logger.info("SELECTED THRESHOLD")
    logger.info("-"*70)
    logger.info(f"Method: {selected_method}")
    logger.info(f"Threshold: {selected_threshold:.6f}")
    logger.info(f"Anomaly rate: {selected_eval['anomaly_rate']:.2%}")
    if 'f1_score' in selected_eval:
        logger.info(f"F1-score: {selected_eval['f1_score']:.4f}")
        logger.info(f"Precision: {selected_eval['precision']:.4f}")
        logger.info(f"Recall: {selected_eval['recall']:.4f}")
    
    # Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'selected_method': selected_method,
        'selected_threshold': float(selected_threshold),
        'selected_evaluation': selected_eval,
        'all_thresholds': {k: float(v) for k, v in thresholds.items()},
        'all_evaluations': {k: {key: val for key, val in v.items()} for k, v in evaluations.items()},
        'error_statistics': {
            'mean': float(errors.mean()),
            'std': float(errors.std()),
            'min': float(errors.min()),
            'max': float(errors.max()),
            'median': float(np.median(errors))
        }
    }
    
    # Save JSON
    results_path = save_dir / "autoencoder_threshold.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved threshold results to {results_path}")
    
    # Plot distribution
    plot_path = save_dir / "autoencoder_threshold_distribution.png"
    plot_threshold_distribution(errors, selected_threshold, selected_method, plot_path)
    
    return results


def select_threshold_logbert(model_path: str,
                            validation_templates: List[str],
                            tokenizer,
                            labels: Optional[np.ndarray] = None,
                            device: str = "cpu",
                            max_length: int = 128,
                            save_dir: str = "output/thresholds") -> Dict:
    """
    Select threshold cho LogBERT
    
    Args:
        model_path: Đường dẫn đến model checkpoint
        validation_templates: Validation templates
        tokenizer: BERT tokenizer
        labels: Ground truth labels (optional)
        device: Device
        max_length: Max sequence length
        save_dir: Directory để lưu results
    
    Returns:
        Dict với selected threshold và evaluation results
    """
    logger.info("\n" + "="*70)
    logger.info("THRESHOLD SELECTION - LOGBERT")
    logger.info("="*70)
    
    # Load model
    model, config = load_logbert_model(model_path, device)
    
    # Calculate anomaly scores
    logger.info("Calculating anomaly scores on validation set...")
    scores = calculate_anomaly_scores(
        model, validation_templates, tokenizer, device, max_length=max_length
    )
    
    logger.info(f"Anomaly scores: mean={scores.mean():.6f}, std={scores.std():.6f}")
    logger.info(f"Min={scores.min():.6f}, Max={scores.max():.6f}")
    
    # Test các threshold methods
    thresholds = {}
    evaluations = {}
    
    # 1. Percentile-based thresholds
    logger.info("\nTesting percentile-based thresholds...")
    for percentile in [90, 95, 99, 99.5, 99.9]:
        threshold = percentile_threshold(scores, percentile)
        thresholds[f'percentile_{percentile}'] = threshold
        eval_result = evaluate_threshold(scores, threshold, labels)
        evaluations[f'percentile_{percentile}'] = eval_result
        logger.info(f"  {percentile}th percentile: threshold={threshold:.6f}, "
                   f"anomaly_rate={eval_result['anomaly_rate']:.2%}")
    
    # 2. Standard deviation thresholds
    logger.info("\nTesting standard deviation thresholds...")
    for n_std in [1.5, 2.0, 2.5, 3.0]:
        threshold = std_threshold(scores, n_std)
        thresholds[f'std_{n_std}'] = threshold
        eval_result = evaluate_threshold(scores, threshold, labels)
        evaluations[f'std_{n_std}'] = eval_result
        logger.info(f"  Mean + {n_std}*std: threshold={threshold:.6f}, "
                   f"anomaly_rate={eval_result['anomaly_rate']:.2%}")
    
    # 3. Fixed thresholds (cho classification)
    logger.info("\nTesting fixed thresholds...")
    for fixed_thresh in [0.3, 0.5, 0.7, 0.9]:
        threshold = fixed_thresh
        thresholds[f'fixed_{fixed_thresh}'] = threshold
        eval_result = evaluate_threshold(scores, threshold, labels)
        evaluations[f'fixed_{fixed_thresh}'] = eval_result
        logger.info(f"  Fixed {fixed_thresh}: anomaly_rate={eval_result['anomaly_rate']:.2%}")
    
    # 4. Optimal threshold (nếu có labels)
    if labels is not None and len(labels) > 0:
        logger.info("\nFinding optimal threshold based on F1-score...")
        optimal_thresh, best_f1 = optimal_threshold_f1(scores, labels)
        if optimal_thresh is not None:
            thresholds['optimal_f1'] = optimal_thresh
            eval_result = evaluate_threshold(scores, optimal_thresh, labels)
            evaluations['optimal_f1'] = eval_result
            logger.info(f"  Optimal (F1): threshold={optimal_thresh:.6f}, "
                       f"F1={best_f1:.4f}, precision={eval_result['precision']:.4f}, "
                       f"recall={eval_result['recall']:.4f}")
    
    # Chọn threshold tốt nhất
    if labels is not None and len(labels) > 0:
        # Chọn dựa trên F1-score
        best_method = max(evaluations.items(), key=lambda x: x[1].get('f1_score', 0))
    else:
        # Chọn dựa trên percentile 95 hoặc fixed 0.5
        if 'percentile_95' in evaluations:
            best_method = ('percentile_95', evaluations['percentile_95'])
        else:
            best_method = ('fixed_0.5', evaluations['fixed_0.5'])
    
    selected_method = best_method[0]
    selected_threshold = thresholds[selected_method]
    selected_eval = best_method[1]
    
    logger.info("\n" + "-"*70)
    logger.info("SELECTED THRESHOLD")
    logger.info("-"*70)
    logger.info(f"Method: {selected_method}")
    logger.info(f"Threshold: {selected_threshold:.6f}")
    logger.info(f"Anomaly rate: {selected_eval['anomaly_rate']:.2%}")
    if 'f1_score' in selected_eval:
        logger.info(f"F1-score: {selected_eval['f1_score']:.4f}")
        logger.info(f"Precision: {selected_eval['precision']:.4f}")
        logger.info(f"Recall: {selected_eval['recall']:.4f}")
    
    # Save results
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'selected_method': selected_method,
        'selected_threshold': float(selected_threshold),
        'selected_evaluation': selected_eval,
        'all_thresholds': {k: float(v) for k, v in thresholds.items()},
        'all_evaluations': {k: {key: val for key, val in v.items()} for k, v in evaluations.items()},
        'score_statistics': {
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'median': float(np.median(scores))
        }
    }
    
    # Save JSON
    results_path = save_dir / "logbert_threshold.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved threshold results to {results_path}")
    
    # Plot distribution
    plot_path = save_dir / "logbert_threshold_distribution.png"
    plot_threshold_distribution(scores, selected_threshold, selected_method, plot_path)
    
    return results


def plot_threshold_distribution(scores: np.ndarray, threshold: float, method: str, save_path: Path):
    """Plot distribution của scores với threshold"""
    plt.figure(figsize=(12, 5))
    
    # Distribution plot
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({method})')
    plt.axvline(np.mean(scores), color='g', linestyle='--', linewidth=1, label='Mean')
    plt.xlabel('Score / Error')
    plt.ylabel('Frequency')
    plt.title('Score/Error Distribution with Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(1, 2, 2)
    sorted_scores = np.sort(scores)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    plt.plot(sorted_scores, cumulative, linewidth=2)
    plt.axvline(threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({method})')
    plt.xlabel('Score / Error')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved threshold distribution plot to {save_path}")
    plt.close()


def select_threshold_autoencoder_local(dataset_name: str = "HDFS",
                                      checkpoint_dir: str = "output/checkpoints",
                                      log_file: str = None,
                                      device: str = "cpu",
                                      save_dir: str = "output/thresholds") -> Dict:
    """
    Select threshold cho Autoencoder từ local checkpoint
    
    Args:
        dataset_name: "HDFS" hoặc "BGL"
        checkpoint_dir: Directory chứa checkpoints
        log_file: Log file để load validation data (nếu None, tự động tìm)
        device: Device
        save_dir: Directory để lưu results
    """
    # Tìm checkpoint
    checkpoint_path = Path(checkpoint_dir) / f"autoencoder_{dataset_name.lower()}_local" / "best_model.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Tìm log file
    if log_file is None:
        base_dir = Path(__file__).parent.parent.parent
        log_file = base_dir / "datasets" / f"{dataset_name}_2k.log"
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
    
    log_file = str(log_file)
    
    # Load và process logs để lấy validation embeddings
    logger.info("Loading validation data...")
    parser = LogParser(persistence_dir=f"drain3_state_{dataset_name.lower()}")
    parsed_logs = parser.parse_dataset(log_file, dataset_type=dataset_name)
    
    # Create preprocessing pipeline
    pipeline = LogPreprocessingPipeline(
        tokenizer_method="word",
        embedder_method="word2vec",
        embedding_dim=128,
        vocab_size=5000
    )
    pipeline.fit(parsed_logs, dataset_type=dataset_name)
    processed_logs = pipeline.process_parsed_logs(parsed_logs)
    
    # Extract embeddings (chỉ lấy normal logs)
    embeddings = []
    for log in processed_logs:
        if log.get('embedded') is not None:
            template = log.get('template', '').upper()
            if not any(keyword in template for keyword in ['ERROR', 'FATAL', 'EXCEPTION', 'FAILED']):
                embeddings.append(log['embedded'])
    
    embeddings = np.array(embeddings)
    
    # Split validation (lấy 20% cuối)
    n_val = int(len(embeddings) * 0.2)
    val_embeddings = embeddings[-n_val:]
    
    logger.info(f"Validation embeddings: {len(val_embeddings)}")
    
    # Select threshold
    results = select_threshold_autoencoder(
        str(checkpoint_path),
        val_embeddings,
        labels=None,  # Unsupervised, không có labels
        device=device,
        save_dir=save_dir
    )
    
    return results


def select_threshold_logbert_local(dataset_name: str = "HDFS",
                                   checkpoint_dir: str = "output/checkpoints",
                                   log_file: str = None,
                                   bert_model_name: str = "distilbert-base-uncased",
                                   device: str = "cpu",
                                   save_dir: str = "output/thresholds") -> Dict:
    """
    Select threshold cho LogBERT từ local checkpoint
    
    Args:
        dataset_name: "HDFS" hoặc "BGL"
        checkpoint_dir: Directory chứa checkpoints
        log_file: Log file để load validation data (nếu None, tự động tìm)
        bert_model_name: BERT model name
        device: Device
        save_dir: Directory để lưu results
    """
    # Tìm checkpoint
    checkpoint_path = Path(checkpoint_dir) / f"logbert_{dataset_name.lower()}_local" / "best_model.pt"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Tìm log file
    if log_file is None:
        base_dir = Path(__file__).parent.parent.parent
        log_file = base_dir / "datasets" / f"{dataset_name}_2k.log"
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
    
    log_file = str(log_file)
    
    # Load và process logs để lấy validation templates
    logger.info("Loading validation data...")
    parser = LogParser(persistence_dir=f"drain3_state_{dataset_name.lower()}")
    parsed_logs = parser.parse_dataset(log_file, dataset_type=dataset_name)
    
    # Extract templates (chỉ lấy normal logs)
    templates = []
    for log in parsed_logs:
        if log.get('template'):
            template = log['template'].upper()
            if not any(keyword in template for keyword in ['ERROR', 'FATAL', 'EXCEPTION', 'FAILED']):
                templates.append(log['template'])
    
    # Split validation (lấy 20% cuối)
    n_val = int(len(templates) * 0.2)
    val_templates = templates[-n_val:]
    
    logger.info(f"Validation templates: {len(val_templates)}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    
    # Select threshold
    results = select_threshold_logbert(
        str(checkpoint_path),
        val_templates,
        tokenizer,
        labels=None,  # Unsupervised, không có labels
        device=device,
        save_dir=save_dir
    )
    
    return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Select threshold for anomaly detection')
    parser.add_argument('--model_type', type=str, required=True, choices=['autoencoder', 'logbert'],
                       help='Model type: autoencoder or logbert')
    parser.add_argument('--dataset', type=str, default='HDFS', choices=['HDFS', 'BGL'],
                       help='Dataset name')
    parser.add_argument('--checkpoint_dir', type=str, default="output/checkpoints",
                       help='Directory containing checkpoints')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to log file (auto-detect if None)')
    parser.add_argument('--bert_model', type=str, default='distilbert-base-uncased',
                       help='BERT model name (for LogBERT)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: cpu or cuda')
    parser.add_argument('--save_dir', type=str, default="output/thresholds",
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    if args.model_type == 'autoencoder':
        results = select_threshold_autoencoder_local(
            dataset_name=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            log_file=args.log_file,
            device=args.device,
            save_dir=args.save_dir
        )
    else:
        results = select_threshold_logbert_local(
            dataset_name=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            log_file=args.log_file,
            bert_model_name=args.bert_model,
            device=args.device,
            save_dir=args.save_dir
        )
    
    logger.info("\n✅ Threshold selection completed!")
    logger.info(f"Selected threshold: {results['selected_threshold']:.6f}")
    logger.info(f"Method: {results['selected_method']}")


if __name__ == "__main__":
    main()

