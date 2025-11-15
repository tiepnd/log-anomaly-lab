# Code Repository - Log Anomaly Detection System

> Complete implementation of AI-based log anomaly detection system using Autoencoder and LogBERT models

---

## 📋 Mục Lục

1. [Tổng Quan](#-tổng-quan)
2. [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
3. [Quick Start](#-quick-start)
4. [Modules](#-modules)
5. [Workflow](#-workflow)
6. [Dependencies](#-dependencies)
7. [Troubleshooting](#-troubleshooting)

---

## 🎯 Tổng Quan

Repository này chứa toàn bộ code implementation cho hệ thống phát hiện bất thường trong log sử dụng AI, bao gồm:

- **Preprocessing**: Log parsing, tokenization, và embedding
- **Training**: Model training, hyperparameter tuning, và threshold selection
- **Evaluation**: Metrics calculation và visualization
- **Deployment**: Realtime pipeline với Kafka, Docker, và microservices

### Models

- **Autoencoder**: Baseline model cho unsupervised anomaly detection
- **LogBERT**: Transformer-based model cho log anomaly detection

### Datasets

- **HDFS**: Hadoop Distributed File System logs (~11M entries)
- **BGL**: Blue Gene/L supercomputer logs (~4.7M entries)

---

## 📁 Cấu Trúc Dự Án

```
code/
├── README.md                    # File này
│
├── preprocessing/               # Module 1: Data Preprocessing
│   ├── README.md
│   ├── core/                   # Core modules (parser, tokenizer, embedder, pipeline)
│   ├── scripts/                # Scripts (test, parse full dataset)
│   └── output/                 # Output files (parsed logs, embeddings)
│
├── training/                    # Module 2: Model Training
│   ├── README.md
│   ├── core/                   # Core modules (datasets, data_loader, training_utils, model_loader)
│   ├── scripts/                # Scripts (train, test, tune, threshold, plot)
│   └── output/                 # Output files (checkpoints, thresholds, tuning results)
│
├── evaluation/                  # Module 3: Model Evaluation
│   ├── README.md
│   ├── core/                   # Core modules (model_loader, metrics, label_loader, plotting)
│   ├── scripts/                # Scripts (evaluate, plot, generate_chapter3_results)
│   └── output/                 # Output files (evaluation results, figures)
│
├── deployment/                  # Module 4: System Deployment
│   ├── README.md
│   ├── docker-compose.yml      # Docker Compose configuration
│   ├── Dockerfile              # Docker image definition
│   ├── config/                 # Configuration files
│   └── src/                    # Source code (services, dashboard)
│
├── models/                      # Model implementations
│   ├── autoencoder.py          # Autoencoder model
│   └── logbert.py              # LogBERT model
│
└── venv/                        # Virtual environment (gitignored)
```

---

## 🚀 Quick Start

### Bước 1: Setup Môi Trường

```bash
# Clone repository (nếu chưa có)
cd "master's thesis/code"

# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Bước 2: Preprocessing

```bash
cd preprocessing
python3 scripts/test_preprocessing.py --section all --log_file ../datasets/HDFS_2k.log
```

### Bước 3: Training

```bash
cd training
python3 scripts/train.py --model autoencoder --local --dataset HDFS --epochs 50
python3 scripts/train.py --model logbert --dataset HDFS --epochs 3
```

### Bước 4: Evaluation

```bash
cd evaluation
python3 scripts/evaluate.py --model_type autoencoder --dataset HDFS
python3 scripts/plot.py --plot_type all --model_type both
```

---

## 📦 Modules

### 1. Preprocessing (`preprocessing/`)

**Mục đích**: Parse raw logs, tokenize, và tạo embeddings

**Core Components**:
- `core/parser.py`: Log parsing với Drain3
- `core/tokenizer.py`: Word-level và BERT tokenization
- `core/embedder.py`: TF-IDF, Word2Vec, FastText, BERT embeddings
- `core/pipeline.py`: End-to-end preprocessing pipeline

**Usage**:
```bash
cd preprocessing
python3 scripts/test_preprocessing.py --section parsing --log_file ../datasets/HDFS_2k.log
```

**Output**: Parsed logs, tokenized sequences, embeddings

**Documentation**: Xem `preprocessing/README.md`

---

### 2. Training (`training/`)

**Mục đích**: Train Autoencoder và LogBERT models

**Core Components**:
- `core/datasets.py`: Dataset classes
- `core/data_loader.py`: Data loading utilities
- `core/training_utils.py`: Training loops và plotting
- `core/model_loader.py`: Model loading utilities

**Scripts**:
- `scripts/train.py`: Unified training script
- `scripts/test.py`: Model testing
- `scripts/tune.py`: Hyperparameter tuning
- `scripts/threshold.py`: Threshold selection
- `scripts/plot_tuning.py`: Visualize tuning results

**Usage**:
```bash
cd training
# Train Autoencoder
python3 scripts/train.py --model autoencoder --local --dataset HDFS --epochs 50

# Train LogBERT
python3 scripts/train.py --model logbert --dataset HDFS --epochs 3

# Tune hyperparameters
python3 scripts/tune.py --model autoencoder --dataset HDFS

# Select threshold
python3 scripts/threshold.py --model_type autoencoder --dataset HDFS
```

**Output**: Model checkpoints, training history, thresholds, tuning results

**Documentation**: Xem `training/README.md`

---

### 3. Evaluation (`evaluation/`)

**Mục đích**: Evaluate models với metrics và visualizations

**Core Components**:
- `core/model_loader.py`: Model loading
- `core/metrics.py`: Metrics calculation (Precision, Recall, F1, ROC-AUC, etc.)
- `core/label_loader.py`: Ground truth label loading và mapping
- `core/plotting.py`: Plotting utilities (ROC curves, confusion matrices, loss curves)
- `core/threshold_loader.py`: Threshold loading

**Scripts**:
- `scripts/evaluate.py`: Unified evaluation script
- `scripts/plot.py`: Unified plotting script
- `scripts/generate_chapter3_charts.py`: Generate Chapter 3 charts
- `scripts/generate_chapter3_results.py`: Generate Chapter 3 results

**Usage**:
```bash
cd evaluation
# Evaluate với ground truth labels
python3 scripts/evaluate.py --model_type autoencoder --dataset HDFS

# Plot all results
python3 scripts/plot.py --plot_type all --model_type both --dataset both
```

**Output**: Evaluation results (JSON), ROC curves, confusion matrices, loss curves

**Documentation**: Xem `evaluation/README.md`

---

### 4. Deployment (`deployment/`)

**Mục đích**: Deploy realtime anomaly detection pipeline

**Components**:
- `docker-compose.yml`: Docker Compose configuration
- `src/log_producer.py`: Log producer service
- `src/preprocessor.py`: Log preprocessor service
- `src/model_service.py`: Model inference service
- `src/alert_service.py`: Alert service
- `src/dashboard/`: Web dashboard

**Usage**:
```bash
cd deployment
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Output**: Realtime anomaly detection pipeline với Kafka, services, và dashboard

**Documentation**: Xem `deployment/README.md`

---

## 🔄 Workflow

### Complete Pipeline

```
1. Preprocessing
   └─> Parse logs → Tokenize → Embed
   
2. Training
   └─> Train models → Tune hyperparameters → Select thresholds
   
3. Evaluation
   └─> Evaluate models → Calculate metrics → Generate visualizations
   
4. Deployment
   └─> Deploy services → Run realtime pipeline → Monitor dashboard
```

### Recommended Workflow

1. **Local Testing** (Small datasets: `HDFS_2k.log`, `BGL_2k.log`)
   ```bash
   # Preprocessing
   cd preprocessing && python3 scripts/test_preprocessing.py --section all
   
   # Training
   cd ../training && python3 scripts/train.py --model autoencoder --local --dataset HDFS
   
   # Evaluation
   cd ../evaluation && python3 scripts/evaluate.py --model_type autoencoder --dataset HDFS
   ```

2. **Full Dataset** (Colab hoặc server với GPU)
   ```bash
   # Preprocessing (full dataset)
   python3 scripts/test_preprocessing.py --section all --parse_full
   
   # Training (full dataset)
   python3 scripts/train.py --model autoencoder --input_file ../preprocessing/output/processed/hdfs_processed.json
   
   # Evaluation (full dataset)
   python3 scripts/evaluate.py --model_type autoencoder --dataset HDFS
   ```

3. **Production Deployment**
   ```bash
   cd deployment
   docker-compose up -d
   ```

---

## 📚 Dependencies

### Core Dependencies

```txt
torch>=1.12.0
transformers>=4.20.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.64.0
```

### Preprocessing Dependencies

```txt
drain3>=0.9.2
gensim>=4.0.0
```

### Deployment Dependencies

```txt
kafka-python>=2.0.2
flask>=2.0.0
fastapi>=0.75.0
docker>=5.0.0
docker-compose>=1.29.0
```

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install by module
pip install -r preprocessing/requirements.txt
pip install -r training/requirements.txt
pip install -r evaluation/requirements.txt
pip install -r deployment/requirements.txt
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError

**Error**: `ModuleNotFoundError: No module named 'preprocessing'`

**Solution**:
```bash
# Đảm bảo đang ở đúng thư mục
cd code/preprocessing
source ../venv/bin/activate
python3 scripts/test_preprocessing.py
```

#### 2. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Giảm batch size: `--batch_size 8` hoặc `--batch_size 4`
- Dùng CPU: `--device cpu`
- Dùng DistilBERT thay vì BERT: `--bert_model distilbert-base-uncased`

#### 3. File Not Found

**Error**: `FileNotFoundError: Checkpoint not found`

**Solution**:
- Kiểm tra paths: `../training/output/checkpoints/autoencoder_hdfs_local/best_model.pt`
- Đảm bảo đã train model trước khi evaluate

#### 4. Import Errors

**Error**: `ImportError: cannot import name 'X' from 'Y'`

**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python version (requires Python 3.8+)
python3 --version
```

#### 5. Docker Issues

**Error**: `docker-compose: command not found`

**Solution**:
```bash
# Install Docker Compose
pip install docker-compose

# Or use docker compose (newer versions)
docker compose up -d
```

---

## 📊 Datasets

### HDFS Dataset

- **File**: `datasets/HDFS.log` (full) hoặc `datasets/HDFS_2k.log` (local)
- **Size**: ~11M entries (full) hoặc ~2000 entries (local)
- **Format**: `081109 203518 143 INFO dfs.DataNode: ...`
- **Labels**: `datasets/HDFS_v1/preprocessed/anomaly_label.csv`

### BGL Dataset

- **File**: `datasets/BGL.log` (full) hoặc `datasets/BGL_2k.log` (local)
- **Size**: ~4.7M entries (full) hoặc ~2000 entries (local)
- **Format**: `- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 ... RAS KERNEL INFO ...`

---

## 🎯 Next Steps

1. ✅ **Preprocessing** - Parse và embed logs
2. ✅ **Training** - Train Autoencoder và LogBERT
3. ✅ **Evaluation** - Evaluate với metrics và visualizations
4. ⏭️ **Deployment** - Deploy realtime pipeline
5. ⏭️ **Monitoring** - Monitor system performance

---

## 📖 Documentation

- **Preprocessing**: `preprocessing/README.md`
- **Training**: `training/README.md`
- **Evaluation**: `evaluation/README.md`
- **Deployment**: `deployment/README.md`

---

## 💡 Tips

1. **Local Testing**: Luôn test với small datasets (`*_2k.log`) trước khi chạy full dataset
2. **GPU Usage**: Sử dụng GPU cho training và evaluation nếu có (cần ~8GB VRAM cho BERT)
3. **Memory Management**: Giảm batch size nếu gặp out-of-memory errors
4. **Checkpoints**: Models được tự động save, có thể resume training từ checkpoint
5. **Logging**: Tất cả modules đều có logging, check logs để debug

---

## 🔗 Related Files

- **Thesis Chapters**: `chapters/chapter_02/`, `chapters/chapter_03/`
- **Figures**: `figures/chapter_02/`, `figures/chapter_03/`
- **Tables**: `tables/chapter_02/`, `tables/chapter_03/`
- **References**: `references/references.bib`

---

## 📝 License

This code is part of a Master's thesis project. All rights reserved.

---

## 👤 Author

Master's Thesis - Nguyen Duc Tiep

---

**Happy Coding! 🚀**
