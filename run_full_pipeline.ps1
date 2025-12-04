# PowerShell script to run the complete pipeline for log anomaly detection
# Usage: .\run_full_pipeline.ps1 [-SkipPreprocessing] [-SkipTraining] [-SkipTuning] [-SkipEvaluation]

param(
    [switch]$SkipPreprocessing,
    [switch]$SkipTraining,
    [switch]$SkipTuning,
    [switch]$SkipEvaluation
)

# Set error action
$ErrorActionPreference = "Stop"

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
    Write-Success "✓ Virtual environment activated"
} else {
    Write-Error "✗ Virtual environment not found. Please create it first."
    exit 1
}

# Check GPU
Write-Warning "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Warning "PyTorch not installed. Installing dependencies..."
    pip install -r requirements.txt
}

# Function to run preprocessing
function Run-Preprocessing {
    Write-Host ""
    Write-Success "=== GIAI ĐOẠN 2: PREPROCESSING ==="
    
    Set-Location preprocessing
    
    Write-Warning "Step 2.1: Parsing HDFS dataset..."
    python scripts\test_preprocessing.py --parse_full --dataset HDFS
    
    Write-Warning "Step 2.2: Parsing BGL dataset..."
    python scripts\test_preprocessing.py --parse_full --dataset BGL
    
    Write-Warning "Step 2.3: Tokenization & Embedding for HDFS (Word2Vec)..."
    python scripts\process_full_dataset.py --dataset HDFS --embedding_method word2vec
    
    Write-Warning "Step 2.3b: Tokenization & Embedding for HDFS (BERT)..."
    python scripts\process_full_dataset.py --dataset HDFS --embedding_method bert
    
    Write-Warning "Step 2.4: Tokenization & Embedding for BGL (Word2Vec)..."
    python scripts\process_full_dataset.py --dataset BGL --embedding_method word2vec
    
    Write-Warning "Step 2.4b: Tokenization & Embedding for BGL (BERT)..."
    python scripts\process_full_dataset.py --dataset BGL --embedding_method bert
    
    Set-Location ..
    Write-Success "✓ Preprocessing completed"
}

# Function to run training
function Run-Training {
    Write-Host ""
    Write-Success "=== GIAI ĐOẠN 3: TRAINING ==="
    
    Set-Location training
    
    Write-Warning "Step 3.1: Training Autoencoder on HDFS..."
    python scripts\train.py --model autoencoder --dataset HDFS --epochs 50 --batch_size 32 --device cuda
    
    Write-Warning "Step 3.2: Training Autoencoder on BGL..."
    python scripts\train.py --model autoencoder --dataset BGL --epochs 50 --batch_size 32 --device cuda
    
    Write-Warning "Step 3.3: Training LogBERT on HDFS..."
    python scripts\train.py --model logbert --dataset HDFS --epochs 10 --batch_size 16 --device cuda
    
    Write-Warning "Step 3.4: Training LogBERT on BGL..."
    python scripts\train.py --model logbert --dataset BGL --epochs 10 --batch_size 16 --device cuda
    
    Set-Location ..
    Write-Success "✓ Training completed"
}

# Function to run tuning
function Run-Tuning {
    Write-Host ""
    Write-Success "=== GIAI ĐOẠN 4: HYPERPARAMETER TUNING ==="
    
    Set-Location training
    
    Write-Warning "Step 4.1: Tuning Autoencoder on HDFS..."
    python scripts\tune.py --model autoencoder --dataset HDFS --epochs 10
    
    Write-Warning "Step 4.2: Tuning Autoencoder on BGL..."
    python scripts\tune.py --model autoencoder --dataset BGL --epochs 10
    
    Write-Warning "Step 4.3: Tuning LogBERT on HDFS..."
    python scripts\tune.py --model logbert --dataset HDFS --epochs 5
    
    Write-Warning "Step 4.4: Tuning LogBERT on BGL..."
    python scripts\tune.py --model logbert --dataset BGL --epochs 5
    
    Write-Warning "Step 4.5: Visualizing tuning results..."
    python scripts\plot_tuning.py --model autoencoder --dataset both
    python scripts\plot_tuning.py --model logbert --dataset both
    
    Set-Location ..
    Write-Success "✓ Tuning completed"
}

# Function to run threshold selection
function Run-ThresholdSelection {
    Write-Host ""
    Write-Success "=== GIAI ĐOẠN 5: THRESHOLD SELECTION ==="
    
    Set-Location training
    
    Write-Warning "Selecting thresholds..."
    python scripts\threshold.py --model_type autoencoder --dataset HDFS
    python scripts\threshold.py --model_type autoencoder --dataset BGL
    python scripts\threshold.py --model_type logbert --dataset HDFS
    python scripts\threshold.py --model_type logbert --dataset BGL
    
    Set-Location ..
    Write-Success "✓ Threshold selection completed"
}

# Function to run evaluation
function Run-Evaluation {
    Write-Host ""
    Write-Success "=== GIAI ĐOẠN 6: EVALUATION ==="
    
    Set-Location evaluation
    
    Write-Warning "Step 6.1-6.4: Evaluating all models..."
    python scripts\evaluate.py --model_type autoencoder --dataset HDFS
    python scripts\evaluate.py --model_type autoencoder --dataset BGL
    python scripts\evaluate.py --model_type logbert --dataset HDFS
    python scripts\evaluate.py --model_type logbert --dataset BGL
    
    Write-Warning "Step 6.5: Generating all plots..."
    python scripts\plot.py --plot_type all --model_type both --dataset both
    
    Set-Location ..
    Write-Success "✓ Evaluation completed"
}

# Main execution
Write-Host ""
Write-Success "========================================"
Write-Success "  LOG ANOMALY DETECTION PIPELINE"
Write-Success "========================================"
Write-Host ""

# Run preprocessing
if (-not $SkipPreprocessing) {
    Run-Preprocessing
} else {
    Write-Warning "Skipping preprocessing"
}

# Run training
if (-not $SkipTraining) {
    Run-Training
} else {
    Write-Warning "Skipping training"
}

# Run tuning
if (-not $SkipTuning) {
    Run-Tuning
} else {
    Write-Warning "Skipping tuning"
}

# Run threshold selection
Run-ThresholdSelection

# Run evaluation
if (-not $SkipEvaluation) {
    Run-Evaluation
} else {
    Write-Warning "Skipping evaluation"
}

Write-Host ""
Write-Success "========================================"
Write-Success "  PIPELINE COMPLETED SUCCESSFULLY!"
Write-Success "========================================"
Write-Host ""

