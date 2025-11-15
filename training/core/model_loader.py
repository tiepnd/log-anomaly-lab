"""
Model loading utilities
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
        Loaded model
    """
    from models.autoencoder import create_autoencoder
    
    logger.info(f"Loading Autoencoder model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    input_dim = config.get('input_dim', 128)
    hidden_dims = config.get('hidden_dims', [256, 128, 64])
    latent_dim = config.get('latent_dim', 32)
    
    # Create model
    model = create_autoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    return model


def load_logbert_model(checkpoint_path: str, device: str = "cpu", bert_model: str = "distilbert-base-uncased"):
    """
    Load LogBERT model từ checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        bert_model: BERT model name
    
    Returns:
        Loaded model và tokenizer
    """
    from models.logbert import create_logbert
    from transformers import AutoTokenizer
    
    logger.info(f"Loading LogBERT model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    bert_model_name = config.get('bert_model', bert_model)
    num_classes = config.get('num_classes', 2)
    task = config.get('task', 'classification')
    
    # Create model
    model = create_logbert(
        bert_model=bert_model_name,
        num_classes=num_classes,
        task=task
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    
    logger.info("Model and tokenizer loaded successfully")
    
    return model, tokenizer

