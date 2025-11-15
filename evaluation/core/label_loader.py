"""
Label loading and mapping utilities
"""
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_ground_truth_labels(label_file: str) -> Dict[str, int]:
    """
    Load ground truth labels từ CSV file
    
    Args:
        label_file: Path to label CSV file (format: BlockId,Label)
    
    Returns:
        Dict mapping block_id -> label (0 = Normal, 1 = Anomaly)
    """
    logger.info(f"Loading ground truth labels from {label_file}")
    
    df = pd.read_csv(label_file)
    
    # Map labels: Normal -> 0, Anomaly -> 1
    label_map = {'Normal': 0, 'Anomaly': 1}
    
    labels_dict = {}
    for _, row in df.iterrows():
        block_id = str(row['BlockId']).strip()
        label_str = str(row['Label']).strip()
        label = label_map.get(label_str, 0)  # Default to Normal if unknown
        labels_dict[block_id] = label
    
    normal_count = sum(1 for v in labels_dict.values() if v == 0)
    anomaly_count = sum(1 for v in labels_dict.values() if v == 1)
    
    logger.info(f"Loaded {len(labels_dict)} labels: {normal_count} Normal, {anomaly_count} Anomaly")
    
    return labels_dict


def extract_block_id_from_log(log_line: str) -> Optional[str]:
    """
    Extract block ID từ HDFS log entry
    
    Args:
        log_line: Raw log line
    
    Returns:
        Block ID (e.g., "blk_123456") hoặc None nếu không tìm thấy
    """
    # Pattern: blk_ followed by digits (may be negative)
    pattern = r'blk_(-?\d+)'
    match = re.search(pattern, log_line)
    
    if match:
        block_id = f"blk_{match.group(1)}"
        return block_id
    
    return None


def map_logs_to_labels(parsed_logs: List[Dict], labels_dict: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map log entries với ground truth labels
    
    Args:
        parsed_logs: List of parsed log entries (có 'original_log' field)
        labels_dict: Dict mapping block_id -> label
    
    Returns:
        Tuple (indices, labels) cho các log entries có labels
    """
    indices = []
    labels = []
    
    for idx, log in enumerate(parsed_logs):
        original_log = log.get('original_log', '')
        block_id = extract_block_id_from_log(original_log)
        
        if block_id and block_id in labels_dict:
            indices.append(idx)
            labels.append(labels_dict[block_id])
    
    logger.info(f"Mapped {len(indices)} log entries to labels out of {len(parsed_logs)} total")
    
    return np.array(indices), np.array(labels)

