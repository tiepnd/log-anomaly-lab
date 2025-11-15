"""
Visualize Hyperparameter Tuning Results
Tạo các plots cho tuning results của Autoencoder và LogBERT
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_tuning_results(results_file: str) -> Dict:
    """Load tuning results từ JSON file"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data


def plot_learning_rate_tuning(autoencoder_results: Optional[Dict] = None,
                              logbert_results: Optional[Dict] = None,
                              save_path: Path = None):
    """
    Plot learning rate tuning results
    
    Args:
        autoencoder_results: Autoencoder tuning results
        logbert_results: LogBERT tuning results
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Autoencoder
    if autoencoder_results:
        results = autoencoder_results.get('all_results', [])
        lrs = [r['params']['learning_rate'] for r in results]
        val_losses = [r['best_val_loss'] for r in results]
        
        axes[0].scatter(lrs, val_losses, alpha=0.6, s=100, color='steelblue')
        axes[0].set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Autoencoder: Learning Rate Tuning', fontsize=14, fontweight='bold')
        axes[0].set_xscale('log')
        axes[0].grid(True, alpha=0.3)
    
    # Plot LogBERT
    if logbert_results:
        results = logbert_results.get('all_results', [])
        lrs = [r['params']['learning_rate'] for r in results]
        val_losses = [r['best_val_loss'] for r in results]
        
        axes[1].scatter(lrs, val_losses, alpha=0.6, s=100, color='orange')
        axes[1].set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        axes[1].set_title('LogBERT: Learning Rate Tuning', fontsize=14, fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved learning rate tuning plot to {save_path}")
    plt.close()


def plot_latent_dim_tuning(autoencoder_results: Dict,
                           save_path: Path = None):
    """
    Plot latent dimension tuning results (Autoencoder only)
    
    Args:
        autoencoder_results: Autoencoder tuning results
        save_path: Path to save plot
    """
    results = autoencoder_results.get('all_results', [])
    latent_dims = [r['params']['latent_dim'] for r in results]
    val_losses = [r['best_val_loss'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(latent_dims, val_losses, alpha=0.6, s=100, color='steelblue')
    plt.xlabel('Latent Dimension', fontsize=12, fontweight='bold')
    plt.ylabel('Validation Loss', fontsize=12, fontweight='bold')
    plt.title('Autoencoder: Latent Dimension Tuning', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved latent dimension tuning plot to {save_path}")
    plt.close()


def plot_window_size_tuning(logbert_results: Dict,
                           save_path: Path = None):
    """
    Plot window size (max_length) tuning results (LogBERT only)
    
    Args:
        logbert_results: LogBERT tuning results
        save_path: Path to save plot
    """
    results = logbert_results.get('all_results', [])
    max_lengths = [r['params']['max_length'] for r in results]
    val_losses = [r['best_val_loss'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(max_lengths, val_losses, alpha=0.6, s=100, color='orange')
    plt.xlabel('Window Size (Max Length)', fontsize=12, fontweight='bold')
    plt.ylabel('Validation Loss', fontsize=12, fontweight='bold')
    plt.title('LogBERT: Window Size Tuning', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved window size tuning plot to {save_path}")
    plt.close()


def plot_batch_size_tuning(autoencoder_results: Optional[Dict] = None,
                           logbert_results: Optional[Dict] = None,
                           save_path: Path = None):
    """
    Plot batch size tuning results
    
    Args:
        autoencoder_results: Autoencoder tuning results
        logbert_results: LogBERT tuning results
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Autoencoder
    if autoencoder_results:
        results = autoencoder_results.get('all_results', [])
        batch_sizes = [r['params']['batch_size'] for r in results]
        val_losses = [r['best_val_loss'] for r in results]
        
        axes[0].scatter(batch_sizes, val_losses, alpha=0.6, s=100, color='steelblue')
        axes[0].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Autoencoder: Batch Size Tuning', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
    
    # Plot LogBERT
    if logbert_results:
        results = logbert_results.get('all_results', [])
        batch_sizes = [r['params']['batch_size'] for r in results]
        val_losses = [r['best_val_loss'] for r in results]
        
        axes[1].scatter(batch_sizes, val_losses, alpha=0.6, s=100, color='orange')
        axes[1].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        axes[1].set_title('LogBERT: Batch Size Tuning', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved batch size tuning plot to {save_path}")
    plt.close()


def plot_comparison_summary(autoencoder_results: Optional[Dict] = None,
                           logbert_results: Optional[Dict] = None,
                           save_path: Path = None):
    """
    Plot comparison summary của best configurations
    
    Args:
        autoencoder_results: Autoencoder tuning results
        logbert_results: LogBERT tuning results
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Best validation loss comparison
    models = []
    best_losses = []
    
    if autoencoder_results:
        models.append('Autoencoder')
        best_losses.append(autoencoder_results['best_result']['best_val_loss'])
    
    if logbert_results:
        models.append('LogBERT')
        best_losses.append(logbert_results['best_result']['best_val_loss'])
    
    if models:
        axes[0, 0].bar(models, best_losses, color=['steelblue', 'orange'][:len(models)])
        axes[0, 0].set_ylabel('Best Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Best Validation Loss Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Top 10 configurations comparison
    if autoencoder_results:
        top_10_ae = autoencoder_results['all_results'][:10]
        config_names_ae = [f"AE-{i+1}" for i in range(len(top_10_ae))]
        val_losses_ae = [r['best_val_loss'] for r in top_10_ae]
        
        axes[0, 1].barh(config_names_ae, val_losses_ae, color='steelblue', alpha=0.7)
        axes[0, 1].set_xlabel('Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Autoencoder: Top 10 Configurations', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    if logbert_results:
        top_10_lb = logbert_results['all_results'][:10]
        config_names_lb = [f"LB-{i+1}" for i in range(len(top_10_lb))]
        val_losses_lb = [r['best_val_loss'] for r in top_10_lb]
        
        axes[1, 0].barh(config_names_lb, val_losses_lb, color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Validation Loss', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('LogBERT: Top 10 Configurations', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Training time comparison
    if autoencoder_results and logbert_results:
        ae_time = sum(r['training_time'] for r in autoencoder_results['all_results'])
        lb_time = sum(r['training_time'] for r in logbert_results['all_results'])
        
        axes[1, 1].bar(['Autoencoder', 'LogBERT'], [ae_time, lb_time], 
                      color=['steelblue', 'orange'])
        axes[1, 1].set_ylabel('Total Training Time (seconds)', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Total Training Time Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved comparison summary to {save_path}")
    plt.close()


def plot_all_tuning_results(autoencoder_file: Optional[str] = None,
                            logbert_file: Optional[str] = None,
                            output_dir: str = "../../figures/chapter_02"):
    """
    Plot tất cả tuning results
    
    Args:
        autoencoder_file: Path to Autoencoder tuning results JSON
        logbert_file: Path to LogBERT tuning results JSON
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    autoencoder_results = None
    logbert_results = None
    
    if autoencoder_file and Path(autoencoder_file).exists():
        autoencoder_results = load_tuning_results(autoencoder_file)
        logger.info(f"Loaded Autoencoder results from {autoencoder_file}")
    
    if logbert_file and Path(logbert_file).exists():
        logbert_results = load_tuning_results(logbert_file)
        logger.info(f"Loaded LogBERT results from {logbert_file}")
    
    # Plot learning rate tuning
    plot_learning_rate_tuning(
        autoencoder_results=autoencoder_results,
        logbert_results=logbert_results,
        save_path=output_dir / "hyperparameter_tuning_lr.png"
    )
    
    # Plot latent dimension tuning (Autoencoder only)
    if autoencoder_results:
        plot_latent_dim_tuning(
            autoencoder_results=autoencoder_results,
            save_path=output_dir / "hyperparameter_tuning_latent_dim.png"
        )
    
    # Plot window size tuning (LogBERT only)
    if logbert_results:
        plot_window_size_tuning(
            logbert_results=logbert_results,
            save_path=output_dir / "hyperparameter_tuning_window_size.png"
        )
    
    # Plot batch size tuning
    plot_batch_size_tuning(
        autoencoder_results=autoencoder_results,
        logbert_results=logbert_results,
        save_path=output_dir / "hyperparameter_tuning_batch_size.png"
    )
    
    # Plot comparison summary
    plot_comparison_summary(
        autoencoder_results=autoencoder_results,
        logbert_results=logbert_results,
        save_path=output_dir / "hyperparameter_tuning_comparison.png"
    )
    
    logger.info("\n✅ All tuning plots generated!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot hyperparameter tuning results')
    parser.add_argument('--autoencoder_file', type=str, default=None,
                       help='Path to Autoencoder tuning results JSON')
    parser.add_argument('--logbert_file', type=str, default=None,
                       help='Path to LogBERT tuning results JSON')
    parser.add_argument('--output_dir', type=str, default='../../figures/chapter_02',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Auto-detect files nếu không specify
    if args.autoencoder_file is None:
        training_dir = Path(__file__).parent.parent  # training/
        autoencoder_file = training_dir / "output" / "tuning_results" / "autoencoder" / "tuning_results.json"
        if autoencoder_file.exists():
            args.autoencoder_file = str(autoencoder_file)
    
    if args.logbert_file is None:
        training_dir = Path(__file__).parent.parent  # training/
        logbert_file = training_dir / "output" / "tuning_results" / "logbert" / "tuning_results.json"
        if logbert_file.exists():
            args.logbert_file = str(logbert_file)
    
    # Plot all results
    plot_all_tuning_results(
        autoencoder_file=args.autoencoder_file,
        logbert_file=args.logbert_file,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

