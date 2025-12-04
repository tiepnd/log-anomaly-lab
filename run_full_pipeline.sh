#!/bin/bash
# Script to run the complete pipeline for log anomaly detection
# Usage: ./run_full_pipeline.sh [--skip-preprocessing] [--skip-training] [--skip-tuning] [--skip-evaluation]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${RED}✗ Virtual environment not found. Please create it first.${NC}"
    exit 1
fi

# Check GPU
echo -e "${YELLOW}Checking GPU availability...${NC}"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" || {
    echo -e "${RED}✗ PyTorch not installed. Installing dependencies...${NC}"
    pip install -r requirements.txt
}

# Parse arguments
SKIP_PREPROCESSING=false
SKIP_TRAINING=false
SKIP_TUNING=false
SKIP_EVALUATION=false

for arg in "$@"; do
    case $arg in
        --skip-preprocessing)
            SKIP_PREPROCESSING=true
            ;;
        --skip-training)
            SKIP_TRAINING=true
            ;;
        --skip-tuning)
            SKIP_TUNING=true
            ;;
        --skip-evaluation)
            SKIP_EVALUATION=true
            ;;
    esac
done

# Function to run preprocessing
run_preprocessing() {
    echo -e "\n${GREEN}=== GIAI ĐOẠN 2: PREPROCESSING ===${NC}"
    
    cd preprocessing
    
    echo -e "${YELLOW}Step 2.1: Parsing HDFS dataset...${NC}"
    python3 scripts/test_preprocessing.py --parse_full --dataset HDFS
    
    echo -e "${YELLOW}Step 2.2: Parsing BGL dataset...${NC}"
    python3 scripts/test_preprocessing.py --parse_full --dataset BGL
    
    echo -e "${YELLOW}Step 2.3: Tokenization & Embedding for HDFS (Word2Vec)...${NC}"
    python3 scripts/process_full_dataset.py --dataset HDFS --embedding_method word2vec
    
    echo -e "${YELLOW}Step 2.3b: Tokenization & Embedding for HDFS (BERT)...${NC}"
    python3 scripts/process_full_dataset.py --dataset HDFS --embedding_method bert
    
    echo -e "${YELLOW}Step 2.4: Tokenization & Embedding for BGL (Word2Vec)...${NC}"
    python3 scripts/process_full_dataset.py --dataset BGL --embedding_method word2vec
    
    echo -e "${YELLOW}Step 2.4b: Tokenization & Embedding for BGL (BERT)...${NC}"
    python3 scripts/process_full_dataset.py --dataset BGL --embedding_method bert
    
    cd ..
    echo -e "${GREEN}✓ Preprocessing completed${NC}"
}

# Function to run training
run_training() {
    echo -e "\n${GREEN}=== GIAI ĐOẠN 3: TRAINING ===${NC}"
    
    cd training
    
    echo -e "${YELLOW}Step 3.1: Training Autoencoder on HDFS...${NC}"
    python3 scripts/train.py --model autoencoder --dataset HDFS --epochs 50 --batch_size 32 --device cuda
    
    echo -e "${YELLOW}Step 3.2: Training Autoencoder on BGL...${NC}"
    python3 scripts/train.py --model autoencoder --dataset BGL --epochs 50 --batch_size 32 --device cuda
    
    echo -e "${YELLOW}Step 3.3: Training LogBERT on HDFS...${NC}"
    python3 scripts/train.py --model logbert --dataset HDFS --epochs 10 --batch_size 16 --device cuda
    
    echo -e "${YELLOW}Step 3.4: Training LogBERT on BGL...${NC}"
    python3 scripts/train.py --model logbert --dataset BGL --epochs 10 --batch_size 16 --device cuda
    
    cd ..
    echo -e "${GREEN}✓ Training completed${NC}"
}

# Function to run tuning
run_tuning() {
    echo -e "\n${GREEN}=== GIAI ĐOẠN 4: HYPERPARAMETER TUNING ===${NC}"
    
    cd training
    
    echo -e "${YELLOW}Step 4.1: Tuning Autoencoder on HDFS...${NC}"
    python3 scripts/tune.py --model autoencoder --dataset HDFS --epochs 10
    
    echo -e "${YELLOW}Step 4.2: Tuning Autoencoder on BGL...${NC}"
    python3 scripts/tune.py --model autoencoder --dataset BGL --epochs 10
    
    echo -e "${YELLOW}Step 4.3: Tuning LogBERT on HDFS...${NC}"
    python3 scripts/tune.py --model logbert --dataset HDFS --epochs 5
    
    echo -e "${YELLOW}Step 4.4: Tuning LogBERT on BGL...${NC}"
    python3 scripts/tune.py --model logbert --dataset BGL --epochs 5
    
    echo -e "${YELLOW}Step 4.5: Visualizing tuning results...${NC}"
    python3 scripts/plot_tuning.py --model autoencoder --dataset both
    python3 scripts/plot_tuning.py --model logbert --dataset both
    
    cd ..
    echo -e "${GREEN}✓ Tuning completed${NC}"
}

# Function to run threshold selection
run_threshold_selection() {
    echo -e "\n${GREEN}=== GIAI ĐOẠN 5: THRESHOLD SELECTION ===${NC}"
    
    cd training
    
    echo -e "${YELLOW}Selecting thresholds...${NC}"
    python3 scripts/threshold.py --model_type autoencoder --dataset HDFS
    python3 scripts/threshold.py --model_type autoencoder --dataset BGL
    python3 scripts/threshold.py --model_type logbert --dataset HDFS
    python3 scripts/threshold.py --model_type logbert --dataset BGL
    
    cd ..
    echo -e "${GREEN}✓ Threshold selection completed${NC}"
}

# Function to run evaluation
run_evaluation() {
    echo -e "\n${GREEN}=== GIAI ĐOẠN 6: EVALUATION ===${NC}"
    
    cd evaluation
    
    echo -e "${YELLOW}Step 6.1-6.4: Evaluating all models...${NC}"
    python3 scripts/evaluate.py --model_type autoencoder --dataset HDFS
    python3 scripts/evaluate.py --model_type autoencoder --dataset BGL
    python3 scripts/evaluate.py --model_type logbert --dataset HDFS
    python3 scripts/evaluate.py --model_type logbert --dataset BGL
    
    echo -e "${YELLOW}Step 6.5: Generating all plots...${NC}"
    python3 scripts/plot.py --plot_type all --model_type both --dataset both
    
    cd ..
    echo -e "${GREEN}✓ Evaluation completed${NC}"
}

# Main execution
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  LOG ANOMALY DETECTION PIPELINE${NC}"
echo -e "${GREEN}========================================${NC}"

# Run preprocessing
if [ "$SKIP_PREPROCESSING" = false ]; then
    run_preprocessing
else
    echo -e "${YELLOW}Skipping preprocessing${NC}"
fi

# Run training
if [ "$SKIP_TRAINING" = false ]; then
    run_training
else
    echo -e "${YELLOW}Skipping training${NC}"
fi

# Run tuning
if [ "$SKIP_TUNING" = false ]; then
    run_tuning
else
    echo -e "${YELLOW}Skipping tuning${NC}"
fi

# Run threshold selection
run_threshold_selection

# Run evaluation
if [ "$SKIP_EVALUATION" = false ]; then
    run_evaluation
else
    echo -e "${YELLOW}Skipping evaluation${NC}"
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  PIPELINE COMPLETED SUCCESSFULLY!${NC}"
echo -e "${GREEN}========================================${NC}"

