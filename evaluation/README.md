# Evaluation Module - Hướng Dẫn Sử Dụng

> Evaluation, metrics calculation, và visualization cho Autoencoder và LogBERT models

---

## 📋 Mục Lục

1. [Quick Start](#-quick-start)
2. [Cấu Trúc Module](#-cấu-trúc-module)
3. [Core Modules](#-core-modules)
4. [Evaluation](#-evaluation)
5. [Plotting](#-plotting)
6. [Evaluation với Ground Truth Labels](#-evaluation-với-ground-truth-labels)
7. [Chapter 3 Results](#-chapter-3-results)
8. [Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

### Bước 1: Evaluate Autoencoder

```bash
cd code/evaluation
source ../venv/bin/activate
python3 scripts/evaluate.py --model_type autoencoder --dataset HDFS
```

**Output:** `output/evaluation/autoencoder_evaluation.json`

### Bước 2: Evaluate LogBERT

```bash
python3 scripts/evaluate.py --model_type logbert --dataset HDFS
```

**Output:** `output/evaluation/logbert_evaluation.json`

### Bước 3: Plot Results

```bash
python3 scripts/plot.py --plot_type all --model_type both --dataset both
```

**Output:** `figures/chapter_02/roc_curve_*.png`, `confusion_matrix_*.png`, `*_loss_curve_*.png`

---

## 📁 Cấu Trúc Module

```
evaluation/
├── README.md                    # File này
│
├── core/                        # ✅ Core utilities (shared code)
│   ├── __init__.py
│   ├── model_loader.py         # Model loading functions
│   ├── label_loader.py         # Label loading and mapping
│   ├── metrics.py              # Metrics calculation
│   ├── plotting.py             # Plotting functions
│   └── threshold_loader.py     # Threshold loading
│
├── scripts/                     # ✅ Tất cả scripts để chạy
│   ├── __init__.py
│   ├── evaluate.py             # ✅ Unified evaluation script
│   ├── plot.py                 # ✅ Unified plotting script
│   ├── generate_chapter3_charts.py    # Generate Chapter 3 charts
│   └── generate_chapter3_results.py   # Generate Chapter 3 results
│
└── output/                      # Tất cả output files
    ├── .gitignore
    ├── evaluation/              # Evaluation results JSON
    └── drain3_state/            # Drain3 state files
```

### ✨ Điểm Nổi Bật

- **Unified Scripts**: Một script cho nhiều use cases (`evaluate.py`, `plot.py`)
- **Core Modules**: Shared code được extract vào `core/` để tránh trùng lặp
- **Organized Output**: Tất cả outputs được tổ chức trong `output/`

---

## 🔧 Core Modules

### `core/model_loader.py`

Model loading utilities:

- **`load_autoencoder_model()`**: Load Autoencoder từ checkpoint
- **`load_logbert_model()`**: Load LogBERT từ checkpoint

**Usage:**
```python
from evaluation.core import load_autoencoder_model, load_logbert_model

# Load Autoencoder
model, config = load_autoencoder_model("path/to/checkpoint.pt", device="cpu")

# Load LogBERT
model, config = load_logbert_model("path/to/checkpoint.pt", device="cpu")
```

### `core/label_loader.py`

Label loading and mapping utilities:

- **`load_ground_truth_labels()`**: Load labels từ CSV file
- **`extract_block_id_from_log()`**: Extract block ID từ log entry
- **`map_logs_to_labels()`**: Map log entries với labels

**Usage:**
```python
from evaluation.core import (
    load_ground_truth_labels,
    map_logs_to_labels
)

# Load labels
labels_dict = load_ground_truth_labels("path/to/labels.csv")

# Map logs to labels
indices, labels = map_logs_to_labels(parsed_logs, labels_dict)
```

### `core/metrics.py`

Metrics calculation utilities:

- **`calculate_metrics()`**: Calculate all metrics (Precision, Recall, F1, ROC-AUC, etc.)
- **`get_roc_curve()`**: Calculate ROC curve
- **`get_pr_curve()`**: Calculate Precision-Recall curve

**Usage:**
```python
from evaluation.core import calculate_metrics, get_roc_curve

# Calculate metrics
metrics = calculate_metrics(y_true, y_pred, y_scores)

# Get ROC curve
fpr, tpr, roc_auc = get_roc_curve(y_true, y_scores)
```

### `core/plotting.py`

Plotting utilities:

- **`plot_roc_curve()`**: Plot ROC curve
- **`plot_confusion_matrix()`**: Plot confusion matrix
- **`plot_loss_curve()`**: Plot loss curves
- **`plot_score_distribution()`**: Plot score/error distribution

**Usage:**
```python
from evaluation.core import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_loss_curve
)

# Plot ROC curve
plot_roc_curve(fpr, tpr, roc_auc, "Model Name", save_path)

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, "Model Name", save_path, metrics)
```

### `core/threshold_loader.py`

Threshold loading utilities:

- **`load_threshold()`**: Load threshold từ JSON file

**Usage:**
```python
from evaluation.core import load_threshold

threshold = load_threshold("autoencoder", "HDFS", threshold_dir="path/to/thresholds")
```

---

## 🎯 Evaluation

### Evaluate Autoencoder

```bash
python3 scripts/evaluate.py --model_type autoencoder \
    --dataset HDFS \
    --checkpoint_dir ../training/output/checkpoints \
    --threshold_dir ../training/output/thresholds \
    --device cpu \
    --save_dir output/evaluation
```

**Arguments:**
- `--model_type`: `autoencoder` hoặc `logbert` (required)
- `--dataset`: `HDFS` hoặc `BGL` (default: HDFS)
- `--checkpoint_dir`: Directory chứa checkpoints (default: `../training/output/checkpoints`)
- `--threshold_dir`: Directory chứa thresholds (default: `../training/output/thresholds`)
- `--log_file`: Path to log file (auto-detect nếu None)
- `--label_file`: Path to ground truth labels CSV (auto-detect cho HDFS nếu None)
- `--device`: `cpu` hoặc `cuda` (default: cpu)
- `--save_dir`: Directory để lưu results (default: `output/evaluation`)

**Output Files:**
- `output/evaluation/autoencoder_evaluation.json` - Evaluation results
- `output/evaluation/roc_curve_autoencoder.png` - ROC curve (nếu có labels)
- `output/evaluation/confusion_matrix_autoencoder.png` - Confusion matrix (nếu có labels)

### Evaluate LogBERT

```bash
python3 scripts/evaluate.py --model_type logbert \
    --dataset HDFS \
    --bert_model distilbert-base-uncased \
    --device cpu
```

**Arguments:**
- `--bert_model`: BERT model name (default: `distilbert-base-uncased`)

**Output Files:**
- `output/evaluation/logbert_evaluation.json` - Evaluation results
- `output/evaluation/roc_curve_logbert.png` - ROC curve (nếu có labels)
- `output/evaluation/confusion_matrix_logbert.png` - Confusion matrix (nếu có labels)

---

## 📊 Plotting

### Plot All Results

```bash
python3 scripts/plot.py --plot_type all \
    --model_type both \
    --dataset both \
    --evaluation_dir output/evaluation \
    --figures_dir ../../figures/chapter_02
```

**Arguments:**
- `--plot_type`: `roc`, `cm`, `loss`, hoặc `all` (default: all)
- `--model_type`: `autoencoder`, `logbert`, hoặc `both` (default: both)
- `--dataset`: `HDFS`, `BGL`, hoặc `both` (default: both)
- `--evaluation_dir`: Directory chứa evaluation results (default: `output/evaluation`)
- `--checkpoint_dir`: Directory chứa checkpoints (default: `../training/output/checkpoints`)
- `--figures_dir`: Directory để lưu figures (default: `../../figures/chapter_02`)

**Output Files:**
- `figures/chapter_02/roc_curve_autoencoder_hdfs.png`
- `figures/chapter_02/confusion_matrix_autoencoder_hdfs.png`
- `figures/chapter_02/autoencoder_loss_curve_hdfs.png`
- `figures/chapter_02/roc_curve_logbert_hdfs.png`
- `figures/chapter_02/confusion_matrix_logbert_hdfs.png`
- `figures/chapter_02/logbert_loss_curve_hdfs.png`

---

## 📝 Evaluation với Ground Truth Labels

### HDFS Dataset Labels

- **File**: `datasets/HDFS_v1/preprocessed/anomaly_label.csv`
- **Format**: CSV với columns `BlockId,Label`
- **Labels**: "Normal" hoặc "Anomaly"
- **Total**: ~575,061 labels (558,223 Normal, 16,838 Anomaly)

### Auto-detect Labels (HDFS)

```bash
# Auto-detect labels cho HDFS
python3 scripts/evaluate.py --model_type autoencoder --dataset HDFS
```

### Manual Label File

```bash
# Specify label file manually
python3 scripts/evaluate.py --model_type autoencoder --dataset HDFS \
    --label_file /path/to/labels.csv
```

### Label Mapping Process

1. **Load Labels**: Load labels từ CSV file
2. **Extract Block IDs**: Extract block IDs từ log entries (pattern: `blk_<number>`)
3. **Map Logs**: Map log entries với labels dựa trên block IDs
4. **Filter**: Filter embeddings/templates có labels
5. **Evaluate**: Evaluate với ground truth labels

### Metrics với Labels

Khi có ground truth labels, evaluation sẽ tính:

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **ROC-AUC**: Area Under ROC Curve
- **PR-AUC**: Area Under Precision-Recall Curve
- **Confusion Matrix**: TP, FP, TN, FN

---

## 📈 Chapter 3 Results

### Generate Chapter 3 Charts

```bash
python3 scripts/generate_chapter3_charts.py
```

**Output:**
- `figures/chapter_03/throughput_vs_load.png` - Throughput vs Load chart
- `figures/chapter_03/latency_distribution.png` - Latency Distribution chart

### Generate Chapter 3 Results (Full)

```bash
python3 scripts/generate_chapter3_results.py
```

**Output:**
- `tables/chapter_03/performance_metrics.md` - Performance metrics table
- `tables/chapter_03/model_comparison.md` - Model comparison tables
- Charts (same as above)

---

## 📊 Evaluation Metrics

### Metrics Tính Toán

1. **Precision**: TP / (TP + FP)
   - Tỷ lệ log được dự đoán là anomaly thực sự là anomaly

2. **Recall**: TP / (TP + FN)
   - Tỷ lệ anomaly được phát hiện trong tổng số anomaly thực tế

3. **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
   - Harmonic mean của Precision và Recall

4. **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
   - Tỷ lệ predictions đúng

5. **ROC-AUC**: Area Under ROC Curve
   - Đánh giá khả năng phân loại ở các threshold khác nhau

6. **PR-AUC**: Area Under Precision-Recall Curve
   - Đánh giá performance khi có class imbalance

### Confusion Matrix

- **TP (True Positives)**: Anomaly được phát hiện đúng
- **FP (False Positives)**: Normal bị phân loại nhầm là anomaly
- **TN (True Negatives)**: Normal được phân loại đúng
- **FN (False Negatives)**: Anomaly bị bỏ sót

---

## 🔍 Unsupervised Evaluation

Khi không có ground truth labels (unsupervised learning), evaluation chỉ có thể:

- Tính reconstruction errors / anomaly scores
- Tính prediction statistics (anomaly_count, anomaly_rate)
- Plot score/error distributions

**Để có full metrics, cần:**
- Test set có labels (ground truth)
- Hoặc manual labeling một subset

---

## 🔧 Troubleshooting

### Lỗi: "ModuleNotFoundError: No module named 'evaluation'"

**Giải pháp:**
```bash
# Đảm bảo đang ở đúng thư mục
cd code/evaluation
source ../venv/bin/activate
python3 scripts/evaluate.py --model_type autoencoder --dataset HDFS
```

### Lỗi: "FileNotFoundError: Checkpoint not found"

**Giải pháp:**
- Kiểm tra checkpoint path: `../training/output/checkpoints/autoencoder_hdfs_local/best_model.pt`
- Đảm bảo đã train model trước khi evaluate

### Lỗi: "Threshold file not found"

**Giải pháp:**
- Kiểm tra threshold path: `../training/output/thresholds/autoencoder_threshold.json`
- Đảm bảo đã select threshold trước khi evaluate

### Lỗi: "Label file not found"

**Giải pháp:**
- Kiểm tra label file path: `datasets/HDFS_v1/preprocessed/anomaly_label.csv`
- Hoặc specify label file manually: `--label_file /path/to/labels.csv`

### Lỗi: "CUDA out of memory"

**Giải pháp:**
- Dùng CPU: `--device cpu`
- Giảm batch size trong code (nếu có thể)

---

## 💡 Tips

1. **Local Testing:** Luôn test với `HDFS_2k.log` trước khi evaluate full dataset
2. **Save Results:** Evaluation results được tự động save vào `output/evaluation/`
3. **Labels:** Evaluation với labels cho kết quả chính xác hơn
4. **Plotting:** Dùng `scripts/plot.py` để plot tất cả results cùng lúc
5. **Core Modules:** Sử dụng core modules để tránh code trùng lặp

---

## 🎯 Next Steps

Sau khi evaluation:
1. ✅ **Evaluation metrics** - Hoàn thành
2. ✅ **ROC curves** - Hoàn thành
3. ✅ **Confusion matrices** - Hoàn thành
4. ✅ **Loss curves** - Hoàn thành
5. ⏭️ **Compare results** - So sánh Autoencoder vs LogBERT
6. ⏭️ **Analyze errors** - Phân tích false positives/negatives

---

## 📚 References

- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [ROC Curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix)

---

**Happy Evaluating! 🚀**
