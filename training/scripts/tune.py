"""
Hyperparameter tuning script cho Autoencoder và LogBERT
"""
import sys
from pathlib import Path

# Add paths to sys.path
base_dir = Path(__file__).parent.parent.parent  # code/
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / "preprocessing"))

# Import từ các file tuning cũ (sẽ xóa sau khi test xong)
import importlib.util

# Load tune_autoencoder
spec_ae = importlib.util.spec_from_file_location("tune_autoencoder", Path(__file__).parent / "tune_autoencoder.py")
tune_ae = importlib.util.module_from_spec(spec_ae)
spec_ae.loader.exec_module(tune_ae)

# Load tune_logbert
spec_lb = importlib.util.spec_from_file_location("tune_logbert", Path(__file__).parent / "tune_logbert.py")
tune_lb = importlib.util.module_from_spec(spec_lb)
spec_lb.loader.exec_module(tune_lb)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune Autoencoder or LogBERT hyperparameters')
    parser.add_argument('--model', type=str, required=True, choices=['autoencoder', 'logbert'],
                       help='Model to tune: autoencoder or logbert')
    
    # Common arguments
    parser.add_argument('--dataset', type=str, default='HDFS', choices=['HDFS', 'BGL'],
                       help='Dataset name: HDFS or BGL')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to log file (auto-detect if None)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs per configuration')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device: cpu or cuda')
    parser.add_argument('--save_dir', type=str, default='output/tuning_results',
                       help='Directory to save results')
    
    # LogBERT specific
    parser.add_argument('--bert_model', type=str, default='distilbert-base-uncased',
                       help='BERT model name (for LogBERT)')
    
    args = parser.parse_args()
    
    # Tune
    if args.model == 'autoencoder':
        logger.info("Starting Autoencoder hyperparameter tuning...")
        result = tune_ae.tune_autoencoder_local(
            dataset_name=args.dataset,
            log_file=args.log_file,
            epochs_per_config=args.epochs,
            device=args.device,
            save_dir=args.save_dir
        )
    elif args.model == 'logbert':
        logger.info("Starting LogBERT hyperparameter tuning...")
        result = tune_lb.tune_logbert_local(
            dataset_name=args.dataset,
            log_file=args.log_file,
            bert_model_name=args.bert_model,
            epochs_per_config=args.epochs,
            device=args.device,
            save_dir=args.save_dir
        )
    
    logger.info("\n✅ Hyperparameter tuning completed successfully!")


if __name__ == "__main__":
    main()

