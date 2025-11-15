"""
Threshold loading utilities
"""
import json
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def load_threshold(model_type: str, dataset_name: str, threshold_dir: str = "thresholds") -> Optional[float]:
    """
    Load threshold từ file
    
    Args:
        model_type: "autoencoder" hoặc "logbert"
        dataset_name: "HDFS" hoặc "BGL"
        threshold_dir: Directory chứa threshold files
    
    Returns:
        Threshold value hoặc None nếu không tìm thấy
    """
    threshold_file = Path(threshold_dir) / f"{model_type}_threshold.json"
    
    if not threshold_file.exists():
        logger.warning(f"Threshold file not found: {threshold_file}")
        logger.info("Using default threshold calculation...")
        return None
    
    with open(threshold_file, 'r') as f:
        threshold_data = json.load(f)
    
    threshold = threshold_data['selected_threshold']
    method = threshold_data.get('selected_method', 'unknown')
    
    logger.info(f"Loaded threshold: {threshold:.6f} (method: {method})")
    return threshold

