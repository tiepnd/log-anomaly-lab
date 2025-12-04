#!/usr/bin/env python3
"""
Script to process full dataset: Parse → Tokenize → Embed
Usage:
    python3 scripts/process_full_dataset.py --dataset HDFS --embedding_method word2vec
    python3 scripts/process_full_dataset.py --dataset BGL --embedding_method bert
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add paths
script_dir = Path(__file__).parent.absolute()
preprocessing_dir = script_dir.parent.absolute()
code_dir = preprocessing_dir.parent.absolute()
sys.path.insert(0, str(code_dir))
sys.path.insert(0, str(preprocessing_dir))

from preprocessing.core import LogParser, LogPreprocessingPipeline

def process_full_dataset(dataset_name: str = "HDFS", embedding_method: str = "word2vec"):
    """
    Process full dataset: Parse → Tokenize → Embed
    
    Args:
        dataset_name: "HDFS" or "BGL"
        embedding_method: "word2vec" (for Autoencoder) or "bert" (for LogBERT)
    """
    print(f"\n{'='*70}")
    print(f"PROCESSING FULL {dataset_name} DATASET")
    print(f"Embedding method: {embedding_method}")
    print(f"{'='*70}\n")
    
    # Setup paths
    base_dir = code_dir
    datasets_dir = base_dir / "datasets"
    output_dir = preprocessing_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths
    if dataset_name == "HDFS":
        log_file = datasets_dir / "HDFS_v1" / "HDFS.log"
    else:
        log_file = datasets_dir / f"{dataset_name}.log"
    
    parsed_file = output_dir / "parsed" / f"{dataset_name.lower()}_parsed.json"
    processed_file = output_dir / "processed" / f"{dataset_name.lower()}_processed_{embedding_method}.json"
    
    # Create directories
    (output_dir / "parsed").mkdir(parents=True, exist_ok=True)
    (output_dir / "processed").mkdir(parents=True, exist_ok=True)
    
    # Step 1: Parse (if not already done)
    if not parsed_file.exists():
        print(f"Step 1: Parsing {dataset_name} dataset...")
        parser = LogParser(
            persistence_dir=str(output_dir / "drain3_state")
        )
        
        start_time = datetime.now()
        results = parser.parse_dataset(
            str(log_file),
            dataset_type=dataset_name,
            max_lines=None,  # Parse all
            output_file=str(parsed_file)
        )
        duration = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ Parsing completed in {duration/60:.2f} minutes")
        print(f"  Parsed {len(results):,} entries")
        print(f"  Output: {parsed_file}")
    else:
        print(f"Step 1: Loading parsed data from {parsed_file}...")
        with open(parsed_file, 'r') as f:
            results = json.load(f)
        print(f"✓ Loaded {len(results):,} parsed entries")
    
    # Step 2: Tokenization & Embedding
    print(f"\nStep 2: Tokenization & Embedding ({embedding_method})...")
    
    # Determine tokenizer and embedder methods
    if embedding_method == "word2vec":
        tokenizer_method = "word"
        embedder_method = "word2vec"
        embedding_dim = 128
    elif embedding_method == "bert":
        tokenizer_method = "bert"
        embedder_method = "bert"
        embedding_dim = 768
    else:
        raise ValueError(f"Unknown embedding method: {embedding_method}")
    
    # Initialize pipeline
    pipeline = LogPreprocessingPipeline(
        tokenizer_method=tokenizer_method,
        embedder_method=embedder_method,
        embedding_dim=embedding_dim,
        vocab_size=5000 if tokenizer_method == "word" else None
    )
    
    # Fit pipeline
    print("  Fitting pipeline...")
    templates = [log['template'] for log in results if log.get('template')]
    pipeline.fit(results)
    
    # Process logs
    print(f"  Processing {len(results):,} logs...")
    start_time = datetime.now()
    processed_logs = pipeline.process_parsed_logs(results)
    duration = (datetime.now() - start_time).total_seconds()
    
    # Save processed logs
    print(f"  Saving processed logs...")
    with open(processed_file, 'w') as f:
        json.dump(processed_logs, f, indent=2)
    
    # Statistics
    success_count = sum(1 for r in processed_logs if r.get('embedded') is not None)
    success_rate = (success_count / len(processed_logs)) * 100 if processed_logs else 0
    
    print(f"\n✓ Processing completed in {duration/60:.2f} minutes")
    print(f"  Processed {len(processed_logs):,} logs")
    print(f"  Success rate: {success_rate:.2f}%")
    print(f"  Output: {processed_file}")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Process full dataset: Parse → Tokenize → Embed'
    )
    parser.add_argument('--dataset', type=str, choices=['HDFS', 'BGL'], default='HDFS',
                       help='Dataset name (default: HDFS)')
    parser.add_argument('--embedding_method', type=str, choices=['word2vec', 'bert'], 
                       default='word2vec',
                       help='Embedding method: word2vec (for Autoencoder) or bert (for LogBERT)')
    
    args = parser.parse_args()
    process_full_dataset(args.dataset, args.embedding_method)

if __name__ == "__main__":
    main()

