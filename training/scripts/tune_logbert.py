"""
Hyperparameter tuning cho LogBERT
Grid search để tìm best hyperparameters
"""
import os
import sys
import json
import time
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import logging

# Add paths to sys.path
base_dir = Path(__file__).parent.parent.parent  # code/
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "preprocessing"))

from models.logbert import create_logbert
from training.core import (
    LogDataset,
    load_and_process_logs_logbert,
    train_epoch_logbert,
    validate_logbert
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(train_templates: List[str],
                val_templates: List[str],
                bert_model_name: str,
                learning_rate: float,
                batch_size: int,
                max_length: int,
                epochs: int,
                device: str = "cpu") -> Dict:
    """
    Train một model với config cụ thể
    
    Returns:
        Dict với training results
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    
    # Create datasets (unsupervised - all normal labels = 0)
    train_labels = [0] * len(train_templates)
    val_labels = [0] * len(val_templates)
    
    train_dataset = LogDataset(train_templates, tokenizer, max_length=max_length, labels=train_labels)
    val_dataset = LogDataset(val_templates, tokenizer, max_length=max_length, labels=val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = create_logbert(
        bert_model_name=bert_model_name,
        task="classification",
        num_labels=2,
        use_pretrained=True
    ).to(device)
    
    # Loss and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch_logbert(model, train_loader, optimizer, scheduler, device, "classification")
        
        # Validate
        val_loss = validate_logbert(model, val_loader, device, "classification")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return {
        'best_val_loss': best_val_loss,
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1],
        'history': history
    }


def grid_search_logbert(train_templates: List[str],
                        val_templates: List[str],
                        hyperparameter_grid: Dict,
                        bert_model_name: str = "distilbert-base-uncased",
                        epochs: int = 5,
                        device: str = "cpu",
                        save_dir: str = "output/tuning_results") -> Dict:
    """
    Grid search cho LogBERT hyperparameters
    """
    logger.info("\n" + "="*70)
    logger.info("GRID SEARCH - LOGBERT HYPERPARAMETER TUNING")
    logger.info("="*70)
    
    # Generate all combinations
    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())
    
    all_combinations = list(itertools.product(*param_values))
    
    logger.info(f"Total combinations to test: {len(all_combinations)}")
    logger.info(f"Hyperparameters: {param_names}")
    
    results = []
    best_result = None
    best_val_loss = float('inf')
    
    save_dir = Path(save_dir) / "logbert"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, combination in enumerate(tqdm(all_combinations, desc="Grid Search")):
        params = dict(zip(param_names, combination))
        
        logger.info(f"\n[{i+1}/{len(all_combinations)}] Testing: {params}")
        
        start_time = time.time()
        
        try:
            result = train_model(
                train_templates=train_templates,
                val_templates=val_templates,
                bert_model_name=bert_model_name,
                learning_rate=params['learning_rate'],
                batch_size=params['batch_size'],
                max_length=params['max_length'],
                epochs=epochs,
                device=device
            )
            
            training_time = time.time() - start_time
            
            result.update({
                'params': params,
                'training_time': training_time,
                'combination_id': i + 1
            })
            
            results.append(result)
            
            if result['best_val_loss'] < best_val_loss:
                best_val_loss = result['best_val_loss']
                best_result = result
            
            logger.info(f"  Best Val Loss: {result['best_val_loss']:.6f}, Time: {training_time:.2f}s")
            
        except Exception as e:
            logger.error(f"  Error training with {params}: {str(e)}")
            continue
    
    logger.info("\n" + "="*70)
    logger.info("GRID SEARCH COMPLETED")
    logger.info("="*70)
    
    # Sort results by validation loss
    results_sorted = sorted(results, key=lambda x: x['best_val_loss'])
    
    logger.info(f"\n🏆 Best Hyperparameters:")
    logger.info(f"   Validation Loss: {best_result['best_val_loss']:.6f}")
    for key, value in best_result['params'].items():
        logger.info(f"   {key}: {value}")
    
    # Save results
    results_path = save_dir / "tuning_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'best_result': best_result,
            'all_results': results_sorted,
            'total_combinations': len(all_combinations),
            'tested_combinations': len(results)
        }, f, indent=2)
    
    logger.info(f"\n✅ Tuning results saved to {results_path}")
    
    # Create DataFrame for analysis
    df_data = []
    for r in results:
        row = r['params'].copy()
        row['best_val_loss'] = r['best_val_loss']
        row['final_train_loss'] = r['final_train_loss']
        row['final_val_loss'] = r['final_val_loss']
        row['training_time'] = r['training_time']
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    df_path = save_dir / "tuning_results.csv"
    df.to_csv(df_path, index=False)
    logger.info(f"✅ Tuning results CSV saved to {df_path}")
    
    # Visualize results
    plot_path = save_dir / "tuning_results_visualization.png"
    visualize_tuning_results(results_sorted[:20], plot_path)  # Top 20
    
    return {
        'best_result': best_result,
        'all_results': results_sorted,
        'dataframe': df
    }


def visualize_tuning_results(results: List[Dict], save_path: Path):
    """Visualize tuning results"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    learning_rates = [r['params']['learning_rate'] for r in results]
    max_lengths = [r['params']['max_length'] for r in results]
    batch_sizes = [r['params']['batch_size'] for r in results]
    val_losses = [r['best_val_loss'] for r in results]
    
    # Plot 1: Learning Rate vs Val Loss
    axes[0, 0].scatter(learning_rates, val_losses, alpha=0.6, s=100)
    axes[0, 0].set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Learning Rate vs Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Max Length (Window Size) vs Val Loss
    axes[0, 1].scatter(max_lengths, val_losses, alpha=0.6, s=100, color='orange')
    axes[0, 1].set_xlabel('Max Length (Window Size)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Window Size vs Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Batch Size vs Val Loss
    axes[1, 0].scatter(batch_sizes, val_losses, alpha=0.6, s=100, color='green')
    axes[1, 0].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Batch Size vs Validation Loss', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Top 10 configurations
    top_10 = results[:10]
    config_names = [f"Config {i+1}" for i in range(len(top_10))]
    top_val_losses = [r['best_val_loss'] for r in top_10]
    
    axes[1, 1].barh(config_names, top_val_losses, color='steelblue')
    axes[1, 1].set_xlabel('Validation Loss', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Top 10 Configurations', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved tuning visualization to {save_path}")
    plt.close()


def tune_logbert_local(dataset_name: str = "HDFS",
                      log_file: str = None,
                      bert_model_name: str = "distilbert-base-uncased",
                      epochs_per_config: int = 5,
                      device: str = "cpu",
                      save_dir: str = "output/tuning_results") -> Dict:
    """
    Tune LogBERT hyperparameters trên local dataset
    """
    # Tìm log file
    if log_file is None:
        log_file = base_dir / "datasets" / f"{dataset_name}_2k.log"
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
    
    log_file = str(log_file)
    
    # Load và process logs
    logger.info("Loading and processing logs...")
    templates = load_and_process_logs_logbert(
        log_file,
        dataset_name,
        max_lines=None
    )
    
    # Split train/validation
    n_val = int(len(templates) * 0.2)
    train_templates = templates[:-n_val]
    val_templates = templates[-n_val:]
    
    logger.info(f"Train samples: {len(train_templates)}, Validation samples: {len(val_templates)}")
    
    # Define hyperparameter grid
    hyperparameter_grid = {
        'learning_rate': [1e-5, 2e-5, 5e-5, 1e-4],
        'max_length': [128, 256, 512, 1024],
        'batch_size': [8, 16, 32, 64]
    }
    
    # Grid search
    results = grid_search_logbert(
        train_templates=train_templates,
        val_templates=val_templates,
        hyperparameter_grid=hyperparameter_grid,
        bert_model_name=bert_model_name,
        epochs=epochs_per_config,
        device=device,
        save_dir=save_dir
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune LogBERT hyperparameters')
    parser.add_argument('--dataset', type=str, default='HDFS', choices=['HDFS', 'BGL'],
                       help='Dataset name')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to log file (auto-detect if None)')
    parser.add_argument('--bert_model', type=str, default='distilbert-base-uncased',
                       help='BERT model name')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs per configuration')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: cpu or cuda')
    parser.add_argument('--save_dir', type=str, default="output/tuning_results",
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    result = tune_logbert_local(
        dataset_name=args.dataset,
        log_file=args.log_file,
        bert_model_name=args.bert_model,
        epochs_per_config=args.epochs,
        device=args.device,
        save_dir=args.save_dir
    )
    
    logger.info("\n✅ Hyperparameter tuning completed!")

