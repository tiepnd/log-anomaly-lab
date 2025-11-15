"""
Training script cho Autoencoder và LogBERT
Hỗ trợ cả local dataset và full dataset
"""
import os
import sys
import json
import time
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from datetime import datetime
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add paths to sys.path
base_dir = Path(__file__).parent.parent.parent  # code/
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "preprocessing"))

from models.autoencoder import create_autoencoder
from models.logbert import create_logbert

# Import core utilities
from training.core import (
    EmbeddingDataset,
    LogDataset,
    load_processed_logs,
    load_and_process_logs_autoencoder,
    load_and_process_logs_logbert,
    train_epoch_autoencoder,
    validate_autoencoder,
    train_epoch_logbert,
    validate_logbert,
    plot_training_curves
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_autoencoder(args):
    """Train Autoencoder"""
    # Setup
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    if args.local:
        # Load từ raw log file
        if args.log_file is None:
            log_file = base_dir / "datasets" / f"{args.dataset}_2k.log"
            if not log_file.exists():
                raise FileNotFoundError(f"Log file not found: {log_file}")
            args.log_file = str(log_file)
        
        logger.info(f"Loading logs from {args.log_file}")
        embeddings, pipeline = load_and_process_logs_autoencoder(
            args.log_file,
            args.dataset,
            max_lines=None
        )
        save_dir = Path(args.save_dir) / f"autoencoder_{args.dataset.lower()}_local"
    else:
        # Load từ processed JSON
        if args.input_file is None:
            raise ValueError("--input_file required when --local is False")
        
        logger.info(f"Loading processed embeddings from {args.input_file}")
        embeddings = load_processed_logs(args.input_file, max_samples=args.max_samples)
        save_dir = Path(args.save_dir) / f"autoencoder_{args.dataset.lower()}"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING AUTOENCODER")
    logger.info("="*70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Embedding shape: {embeddings.shape}")
    logger.info(f"Save directory: {save_dir}")
    
    # Split train/validation
    n_samples = len(embeddings)
    n_val = int(n_samples * args.validation_split)
    n_train = n_samples - n_val
    
    train_embeddings = embeddings[:n_train]
    val_embeddings = embeddings[n_train:]
    
    logger.info(f"Train samples: {n_train}, Validation samples: {n_val}")
    
    # Create datasets
    train_dataset = EmbeddingDataset(train_embeddings)
    val_dataset = EmbeddingDataset(val_embeddings)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = create_autoencoder(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    # Training loop
    logger.info("\n" + "-"*70)
    logger.info("STARTING TRAINING")
    logger.info("-"*70)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch_autoencoder(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate_autoencoder(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            
            checkpoint_path = save_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'history': history,
                'config': {
                    'input_dim': args.input_dim,
                    'hidden_dims': args.hidden_dims,
                    'latent_dim': args.latent_dim,
                    'learning_rate': args.lr,
                    'batch_size': args.batch_size
                }
            }, checkpoint_path)
            logger.info(f"✓ Saved best model (val_loss={val_loss:.6f})")
    
    # Training completed
    total_time = time.time() - start_time
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_usage = mem_after - mem_before
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED")
    logger.info("="*70)
    logger.info(f"Total training time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    logger.info(f"Memory usage: {mem_usage:.2f} MB")
    logger.info(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")
    
    # Save final model
    final_path = save_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'input_dim': args.input_dim,
            'hidden_dims': args.hidden_dims,
            'latent_dim': args.latent_dim,
            'learning_rate': args.lr,
            'batch_size': args.batch_size
        },
        'training_stats': {
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'memory_usage_mb': mem_usage,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
    }, final_path)
    
    # Save training history
    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'history': history,
            'training_stats': {
                'total_time_seconds': total_time,
                'total_time_minutes': total_time / 60,
                'memory_usage_mb': mem_usage,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss
            }
        }, f, indent=2)
    
    # Plot training curves
    plot_path = save_dir / "training_curve.png"
    plot_training_curves(history, plot_path)
    
    logger.info(f"\n✅ All outputs saved to: {save_dir}")
    logger.info(f"   - Best model: {save_dir / 'best_model.pt'}")
    logger.info(f"   - Final model: {save_dir / 'final_model.pt'}")
    logger.info(f"   - Training history: {save_dir / 'training_history.json'}")
    logger.info(f"   - Training curve: {save_dir / 'training_curve.png'}")


def train_logbert(args):
    """Train LogBERT"""
    # Setup
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load data
    if args.log_file is None:
        log_file = base_dir / "datasets" / f"{args.dataset}_2k.log"
        if not log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        args.log_file = str(log_file)
    
    logger.info(f"Loading logs from {args.log_file}")
    templates = load_and_process_logs_logbert(
        args.log_file,
        args.dataset,
        max_lines=None
    )
    
    if len(templates) == 0:
        raise ValueError("No templates extracted from logs!")
    
    logger.info(f"Total templates: {len(templates)}")
    
    # Load tokenizer
    logger.info(f"\nLoading tokenizer: {args.bert_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    
    # Split train/validation
    n_samples = len(templates)
    n_val = int(n_samples * args.validation_split)
    n_train = n_samples - n_val
    
    train_templates = templates[:n_train]
    val_templates = templates[n_train:]
    
    logger.info(f"Train samples: {n_train}, Validation samples: {n_val}")
    
    # Create datasets (unsupervised - all normal labels = 0)
    train_labels = [0] * len(train_templates)
    val_labels = [0] * len(val_templates)
    
    train_dataset = LogDataset(train_templates, tokenizer, max_length=args.max_length, labels=train_labels)
    val_dataset = LogDataset(val_templates, tokenizer, max_length=args.max_length, labels=val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = create_logbert(
        bert_model_name=args.bert_model,
        task=args.task,
        num_labels=2,
        use_pretrained=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Loss and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(total_steps * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': [],
        'learning_rates': []
    }
    
    # Training loop
    logger.info("\n" + "-"*70)
    logger.info("STARTING TRAINING")
    logger.info("-"*70)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch_logbert(model, train_loader, optimizer, scheduler, device, args.task)
        
        # Validate
        val_loss = validate_logbert(model, val_loader, device, args.task)
        
        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0] if scheduler else args.lr
        
        logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        logger.info(f"Learning Rate: {current_lr:.2e}, Time: {epoch_time:.2f}s")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
            logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch + 1)
        history['learning_rates'].append(current_lr)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            
            save_dir = Path(args.save_dir) / f"logbert_{args.dataset.lower()}_local"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = save_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'history': history,
                'config': {
                    'bert_model': args.bert_model,
                    'task': args.task,
                    'learning_rate': args.lr,
                    'batch_size': args.batch_size,
                    'max_length': args.max_length
                }
            }, checkpoint_path)
            logger.info(f"✓ Saved best model (val_loss={val_loss:.6f})")
    
    # Training completed
    total_time = time.time() - start_time
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_usage = mem_after - mem_before
    
    gpu_memory_peak = 0
    if torch.cuda.is_available():
        gpu_memory_peak = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED")
    logger.info("="*70)
    logger.info(f"Total training time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    logger.info(f"CPU Memory usage: {mem_usage:.2f} MB")
    if torch.cuda.is_available():
        logger.info(f"GPU Memory peak: {gpu_memory_peak:.2f} GB")
    logger.info(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})")
    
    # Save final model
    save_dir = Path(args.save_dir) / f"logbert_{args.dataset.lower()}_local"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    final_path = save_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'bert_model': args.bert_model,
            'task': args.task,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'max_length': args.max_length
        },
        'training_stats': {
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'memory_usage_mb': mem_usage,
            'gpu_memory_peak_gb': gpu_memory_peak,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }
    }, final_path)
    
    # Save training history
    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'history': history,
            'training_stats': {
                'total_time_seconds': total_time,
                'total_time_minutes': total_time / 60,
                'memory_usage_mb': mem_usage,
                'gpu_memory_peak_gb': gpu_memory_peak,
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss
            }
        }, f, indent=2)
    
    # Plot training curves
    plot_path = save_dir / "training_curve.png"
    plot_training_curves(history, plot_path)
    
    logger.info(f"\n✅ All outputs saved to: {save_dir}")
    logger.info(f"   - Best model: {save_dir / 'best_model.pt'}")
    logger.info(f"   - Final model: {save_dir / 'final_model.pt'}")
    logger.info(f"   - Training history: {save_dir / 'training_history.json'}")
    logger.info(f"   - Training curve: {save_dir / 'training_curve.png'}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Autoencoder or LogBERT')
    parser.add_argument('--model', type=str, required=True, choices=['autoencoder', 'logbert'],
                       help='Model to train: autoencoder or logbert')
    
    # Common arguments
    parser.add_argument('--dataset', type=str, default='HDFS', choices=['HDFS', 'BGL'],
                       help='Dataset name: HDFS or BGL')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: cpu or cuda')
    parser.add_argument('--save_dir', type=str, default='output/checkpoints',
                       help='Directory to save outputs')
    parser.add_argument('--validation_split', type=float, default=0.2,
                       help='Validation split ratio')
    
    # Autoencoder arguments
    parser.add_argument('--local', action='store_true',
                       help='Use local dataset (HDFS_2k.log, BGL_2k.log)')
    parser.add_argument('--input_file', type=str, default=None,
                       help='Path to processed embeddings JSON (required if --local is False)')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to log file (auto-detect if None and --local)')
    parser.add_argument('--input_dim', type=int, default=128,
                       help='Input embedding dimension')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128, 64],
                       help='Hidden dimensions')
    parser.add_argument('--latent_dim', type=int, default=32,
                       help='Latent dimension')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples (for testing)')
    
    # LogBERT arguments
    parser.add_argument('--bert_model', type=str, default='distilbert-base-uncased',
                       help='BERT model name (bert-base-uncased, distilbert-base-uncased)')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'reconstruction'],
                       help='Task: classification or reconstruction')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Max sequence length')
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Warmup steps (0 = auto 10%%)')
    
    args = parser.parse_args()
    
    # Train
    if args.model == 'autoencoder':
        train_autoencoder(args)
    elif args.model == 'logbert':
        train_logbert(args)
    
    logger.info("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()

