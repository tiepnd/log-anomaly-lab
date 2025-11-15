"""
Model loading utilities for evaluation
"""
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_autoencoder_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load Autoencoder model từ checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple (model, config)
    """
    from models.autoencoder import create_autoencoder
    
    logger.info(f"Loading Autoencoder model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = create_autoencoder(
        input_dim=config['input_dim'],
        hidden_dims=config['hidden_dims'],
        latent_dim=config['latent_dim']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("Autoencoder model loaded successfully")
    
    return model, config


def load_logbert_model(checkpoint_path: str, device: str = "cpu"):
    """
    Load LogBERT model từ checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Tuple (model, config)
    """
    from models.logbert import create_logbert
    
    logger.info(f"Loading LogBERT model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = create_logbert(
        bert_model_name=config.get('bert_model', config.get('bert_model_name', 'distilbert-base-uncased')),
        task=config.get('task', 'classification'),
        num_labels=2,
        use_pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("LogBERT model loaded successfully")
    
    return model, config

