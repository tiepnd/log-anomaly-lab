"""
Data loading utilities
"""
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_processed_logs(filepath: str, max_samples: Optional[int] = None) -> np.ndarray:
    """
    Load processed embeddings từ JSON file
    
    Args:
        filepath: Path to processed logs JSON file
        max_samples: Maximum number of samples (None = all)
    
    Returns:
        numpy array of embeddings (N, D)
    """
    logger.info(f"Loading processed logs from {filepath}")
    
    with open(filepath, 'r') as f:
        processed_logs = json.load(f)
    
    embeddings = []
    for log in processed_logs:
        if log.get('embedded') is not None:
            embeddings.append(log['embedded'])
    
    embeddings = np.array(embeddings)
    
    if max_samples is not None and len(embeddings) > max_samples:
        embeddings = embeddings[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    logger.info(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    return embeddings


def load_and_process_logs_autoencoder(log_file: str, dataset_type: str, max_lines: Optional[int] = None) -> Tuple[np.ndarray, object]:
    """
    Load raw logs và process qua pipeline để lấy embeddings cho Autoencoder
    
    Args:
        log_file: Đường dẫn đến file log
        dataset_type: "HDFS" hoặc "BGL"
        max_lines: Số dòng tối đa (None = tất cả)
    
    Returns:
        Tuple (embeddings, pipeline)
    """
    from preprocessing.core.parser import LogParser
    from preprocessing.core.pipeline import LogPreprocessingPipeline
    
    logger.info(f"Loading logs from {log_file}")
    
    # Parse logs
    base_dir = Path(__file__).parent.parent.parent
    persistence_dir = str(base_dir / "preprocessing" / "output" / "drain3_state")
    parser = LogParser(persistence_dir=persistence_dir)
    parsed_logs = parser.parse_dataset(
        log_file,
        dataset_type=dataset_type,
        max_lines=max_lines,
        output_file=None
    )
    
    logger.info(f"Parsed {len(parsed_logs)} logs")
    
    # Create preprocessing pipeline
    pipeline = LogPreprocessingPipeline(
        tokenizer_method="word",
        embedder_method="word2vec",
        embedding_dim=128,
        vocab_size=5000
    )
    
    # Fit pipeline
    logger.info("Fitting preprocessing pipeline...")
    pipeline.fit(parsed_logs, dataset_type=dataset_type)
    
    # Process logs
    logger.info("Processing logs through pipeline...")
    processed_logs = pipeline.process_parsed_logs(parsed_logs)
    
    # Extract embeddings (chỉ lấy normal logs - không có error trong template)
    embeddings = []
    for log in processed_logs:
        if log.get('embedded') is not None:
            # Filter: chỉ lấy normal logs (không có ERROR, FATAL, etc trong template)
            template = log.get('template', '').upper()
            if not any(keyword in template for keyword in ['ERROR', 'FATAL', 'EXCEPTION', 'FAILED']):
                embeddings.append(log['embedded'])
    
    embeddings = np.array(embeddings)
    logger.info(f"Extracted {len(embeddings)} normal log embeddings")
    
    return embeddings, pipeline


def load_and_process_logs_logbert(log_file: str, dataset_type: str, max_lines: Optional[int] = None) -> list:
    """
    Load raw logs và parse để lấy templates cho LogBERT
    
    Args:
        log_file: Đường dẫn đến file log
        dataset_type: "HDFS" hoặc "BGL"
        max_lines: Số dòng tối đa (None = tất cả)
    
    Returns:
        List of templates (normal logs only)
    """
    from preprocessing.core.parser import LogParser
    
    logger.info(f"Loading logs from {log_file}")
    
    # Parse logs
    base_dir = Path(__file__).parent.parent.parent
    persistence_dir = str(base_dir / "preprocessing" / "output" / "drain3_state")
    parser = LogParser(persistence_dir=persistence_dir)
    parsed_logs = parser.parse_dataset(
        log_file,
        dataset_type=dataset_type,
        max_lines=max_lines,
        output_file=None
    )
    
    logger.info(f"Parsed {len(parsed_logs)} logs")
    
    # Extract templates (chỉ lấy normal logs)
    templates = []
    for log in parsed_logs:
        if log.get('template'):
            template = log['template'].upper()
            # Filter: chỉ lấy normal logs (không có ERROR, FATAL, etc.)
            if not any(keyword in template for keyword in ['ERROR', 'FATAL', 'EXCEPTION', 'FAILED']):
                templates.append(log['template'])
    
    logger.info(f"Extracted {len(templates)} normal log templates")
    
    return templates

