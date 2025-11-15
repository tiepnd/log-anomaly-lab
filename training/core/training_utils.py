"""
Training utilities: train_epoch, validate, plotting
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def train_epoch_autoencoder(model, dataloader, optimizer, criterion, device):
    """Train một epoch cho Autoencoder"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = batch.to(device)
        
        reconstructed, _ = model(batch)
        loss = criterion(reconstructed, batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_autoencoder(model, dataloader, criterion, device):
    """Validate Autoencoder"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            batch = batch.to(device)
            
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_epoch_logbert(model, dataloader, optimizer, scheduler, device, task="classification"):
    """Train một epoch cho LogBERT"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch.get('labels', None)
        
        if labels is not None:
            labels = labels.to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask, labels=labels)
        
        # Get loss
        if 'loss' in outputs:
            loss = outputs['loss']
        else:
            # Calculate loss manually if not provided
            if task == "classification":
                # Use cross-entropy loss
                logits = outputs['logits']
                if labels is None:
                    # Create dummy labels (all normal = 0)
                    labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
            else:
                # Reconstruction loss
                reconstructed = outputs['reconstructed']
                original = outputs['original']
                loss_fn = nn.MSELoss()
                loss = loss_fn(reconstructed, original)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_logbert(model, dataloader, device, task="classification"):
    """Validate LogBERT"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch.get('labels', None)
            
            if labels is not None:
                labels = labels.to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, labels=labels)
            
            # Get loss
            if 'loss' in outputs:
                loss = outputs['loss']
            else:
                # Calculate loss manually if not provided
                if task == "classification":
                    logits = outputs['logits']
                    if labels is None:
                        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)
                else:
                    reconstructed = outputs['reconstructed']
                    original = outputs['original']
                    loss_fn = nn.MSELoss()
                    loss = loss_fn(reconstructed, original)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def plot_training_curves(history: dict, save_path: Path):
    """
    Plot training curves
    
    Args:
        history: Dict với keys 'train_loss', 'val_loss', 'epochs'
        save_path: Path để save plot
    """
    epochs = history.get('epochs', list(range(1, len(history['train_loss']) + 1)))
    train_loss = history['train_loss']
    val_loss = history.get('val_loss', [])
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved to {save_path}")

