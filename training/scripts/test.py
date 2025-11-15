"""
Test script cho Autoencoder và LogBERT
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
from transformers import AutoTokenizer

# Add paths to sys.path
base_dir = Path(__file__).parent.parent.parent  # code/
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "preprocessing"))

from models.autoencoder import create_autoencoder
from models.logbert import create_logbert, count_parameters
from training.core import EmbeddingDataset, load_processed_logs
from torch.utils.data import DataLoader

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_autoencoder():
    """Test Autoencoder"""
    print("\n" + "="*70)
    print("TESTING AUTOENCODER")
    print("="*70)
    
    # Test 1: Model creation
    print("\n[1/3] Testing model creation...")
    model = create_autoencoder(
        input_dim=128,
        hidden_dims=[256, 128, 64],
        latent_dim=32
    )
    
    x = torch.randn(10, 128)
    reconstructed, latent = model(x)
    
    print(f"✅ Model created:")
    print(f"   Input shape: {x.shape}")
    print(f"   Latent shape: {latent.shape}")
    print(f"   Reconstructed shape: {reconstructed.shape}")
    
    error = model.get_reconstruction_error(x)
    print(f"   Reconstruction error - Mean: {error.mean().item():.4f}, Std: {error.std().item():.4f}")
    
    # Test 2: Training sample
    print("\n[2/3] Testing training on sample...")
    processed_file = base_dir / "preprocessing" / "output" / "processed" / "hdfs_processed_word2vec.json"
    
    if processed_file.exists():
        embeddings = load_processed_logs(str(processed_file), max_samples=1000)
        
        if len(embeddings) > 0:
            dataset = EmbeddingDataset(embeddings[:100])
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            model.train()
            for batch in dataloader:
                reconstructed, _ = model(batch)
                loss = criterion(reconstructed, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break
            
            print(f"✅ Training test passed (loss: {loss.item():.4f})")
        else:
            print("⚠️  No embeddings found")
    else:
        print(f"⚠️  Processed file not found: {processed_file}")
    
    # Test 3: Save/load
    print("\n[3/3] Testing save/load...")
    save_dir = Path("output/checkpoints/test_autoencoder")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = save_dir / "test_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': 128,
            'hidden_dims': [256, 128, 64],
            'latent_dim': 32
        }
    }, checkpoint_path)
    
    # Load
    checkpoint = torch.load(checkpoint_path)
    model2 = create_autoencoder(
        input_dim=checkpoint['config']['input_dim'],
        hidden_dims=checkpoint['config']['hidden_dims'],
        latent_dim=checkpoint['config']['latent_dim']
    )
    model2.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Save/load test passed")
    
    print("\n✅ All Autoencoder tests passed!")


def test_logbert():
    """Test LogBERT"""
    print("\n" + "="*70)
    print("TESTING LOGBERT")
    print("="*70)
    
    # Test 1: Model creation
    print("\n[1/3] Testing model creation...")
    model = create_logbert(
        bert_model_name="distilbert-base-uncased",
        task="classification",
        num_labels=2,
        use_pretrained=True
    )
    
    print(f"✅ Model created:")
    print(f"   Parameters: {count_parameters(model):,}")
    print(f"   Hidden size: {model.hidden_size}")
    
    # Test 2: Inference
    print("\n[2/3] Testing inference...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model.eval()
    
    sample_logs = [
        "dfs.DataNode: Receiving block blk_123",
        "ERROR: Connection timeout to database",
        "INFO: Service started successfully"
    ]
    
    encoded = tokenizer(
        sample_logs,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(encoded['input_ids'], encoded['attention_mask'])
        logits = outputs['logits']
        scores = torch.softmax(logits, dim=-1)
    
    print(f"✅ Inference test passed")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Sample scores: {scores[0].tolist()}")
    
    # Test 3: Memory usage
    print("\n[3/3] Testing memory usage...")
    if torch.cuda.is_available():
        model = model.cuda()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(encoded['input_ids'].cuda(), encoded['attention_mask'].cuda())
        
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"✅ GPU memory test passed: {gpu_memory:.2f} GB")
    else:
        print("⚠️  CUDA not available, skipping GPU memory test")
    
    print("\n✅ All LogBERT tests passed!")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Autoencoder or LogBERT')
    parser.add_argument('--model', type=str, default='all', choices=['autoencoder', 'logbert', 'all'],
                       help='Model to test: autoencoder, logbert, or all')
    
    args = parser.parse_args()
    
    if args.model == 'autoencoder' or args.model == 'all':
        test_autoencoder()
    
    if args.model == 'logbert' or args.model == 'all':
        test_logbert()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()

