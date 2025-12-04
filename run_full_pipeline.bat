@echo off
REM Batch script to run the complete pipeline for log anomaly detection
REM Usage: run_full_pipeline.bat [--skip-preprocessing] [--skip-training] [--skip-tuning] [--skip-evaluation]

setlocal enabledelayedexpansion

cd /d "%~dp0"

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✓ Virtual environment activated
) else (
    echo ✗ Virtual environment not found. Please create it first.
    exit /b 1
)

REM Check arguments
set SKIP_PREPROCESSING=0
set SKIP_TRAINING=0
set SKIP_TUNING=0
set SKIP_EVALUATION=0

:parse_args
if "%~1"=="" goto :end_parse
if /i "%~1"=="--skip-preprocessing" set SKIP_PREPROCESSING=1
if /i "%~1"=="--skip-training" set SKIP_TRAINING=1
if /i "%~1"=="--skip-tuning" set SKIP_TUNING=1
if /i "%~1"=="--skip-evaluation" set SKIP_EVALUATION=1
shift
goto :parse_args
:end_parse

echo.
echo ========================================
echo   LOG ANOMALY DETECTION PIPELINE
echo ========================================
echo.

REM Run preprocessing
if %SKIP_PREPROCESSING%==0 (
    echo === GIAI ĐOẠN 2: PREPROCESSING ===
    cd preprocessing
    
    echo Step 2.1: Parsing HDFS dataset...
    python scripts\test_preprocessing.py --parse_full --dataset HDFS
    
    echo Step 2.2: Parsing BGL dataset...
    python scripts\test_preprocessing.py --parse_full --dataset BGL
    
    echo Step 2.3: Tokenization ^& Embedding for HDFS (Word2Vec)...
    python scripts\process_full_dataset.py --dataset HDFS --embedding_method word2vec
    
    echo Step 2.3b: Tokenization ^& Embedding for HDFS (BERT)...
    python scripts\process_full_dataset.py --dataset HDFS --embedding_method bert
    
    echo Step 2.4: Tokenization ^& Embedding for BGL (Word2Vec)...
    python scripts\process_full_dataset.py --dataset BGL --embedding_method word2vec
    
    echo Step 2.4b: Tokenization ^& Embedding for BGL (BERT)...
    python scripts\process_full_dataset.py --dataset BGL --embedding_method bert
    
    cd ..
    echo ✓ Preprocessing completed
) else (
    echo Skipping preprocessing
)

REM Run training
if %SKIP_TRAINING%==0 (
    echo.
    echo === GIAI ĐOẠN 3: TRAINING ===
    cd training
    
    echo Step 3.1: Training Autoencoder on HDFS...
    python scripts\train.py --model autoencoder --dataset HDFS --epochs 50 --batch_size 32 --device cuda
    
    echo Step 3.2: Training Autoencoder on BGL...
    python scripts\train.py --model autoencoder --dataset BGL --epochs 50 --batch_size 32 --device cuda
    
    echo Step 3.3: Training LogBERT on HDFS...
    python scripts\train.py --model logbert --dataset HDFS --epochs 10 --batch_size 16 --device cuda
    
    echo Step 3.4: Training LogBERT on BGL...
    python scripts\train.py --model logbert --dataset BGL --epochs 10 --batch_size 16 --device cuda
    
    cd ..
    echo ✓ Training completed
) else (
    echo Skipping training
)

REM Run tuning
if %SKIP_TUNING%==0 (
    echo.
    echo === GIAI ĐOẠN 4: HYPERPARAMETER TUNING ===
    cd training
    
    echo Step 4.1: Tuning Autoencoder on HDFS...
    python scripts\tune.py --model autoencoder --dataset HDFS --epochs 10
    
    echo Step 4.2: Tuning Autoencoder on BGL...
    python scripts\tune.py --model autoencoder --dataset BGL --epochs 10
    
    echo Step 4.3: Tuning LogBERT on HDFS...
    python scripts\tune.py --model logbert --dataset HDFS --epochs 5
    
    echo Step 4.4: Tuning LogBERT on BGL...
    python scripts\tune.py --model logbert --dataset BGL --epochs 5
    
    echo Step 4.5: Visualizing tuning results...
    python scripts\plot_tuning.py --model autoencoder --dataset both
    python scripts\plot_tuning.py --model logbert --dataset both
    
    cd ..
    echo ✓ Tuning completed
) else (
    echo Skipping tuning
)

REM Run threshold selection
echo.
echo === GIAI ĐOẠN 5: THRESHOLD SELECTION ===
cd training

echo Selecting thresholds...
python scripts\threshold.py --model_type autoencoder --dataset HDFS
python scripts\threshold.py --model_type autoencoder --dataset BGL
python scripts\threshold.py --model_type logbert --dataset HDFS
python scripts\threshold.py --model_type logbert --dataset BGL

cd ..
echo ✓ Threshold selection completed

REM Run evaluation
if %SKIP_EVALUATION%==0 (
    echo.
    echo === GIAI ĐOẠN 6: EVALUATION ===
    cd evaluation
    
    echo Step 6.1-6.4: Evaluating all models...
    python scripts\evaluate.py --model_type autoencoder --dataset HDFS
    python scripts\evaluate.py --model_type autoencoder --dataset BGL
    python scripts\evaluate.py --model_type logbert --dataset HDFS
    python scripts\evaluate.py --model_type logbert --dataset BGL
    
    echo Step 6.5: Generating all plots...
    python scripts\plot.py --plot_type all --model_type both --dataset both
    
    cd ..
    echo ✓ Evaluation completed
) else (
    echo Skipping evaluation
)

echo.
echo ========================================
echo   PIPELINE COMPLETED SUCCESSFULLY!
echo ========================================
echo.

endlocal

