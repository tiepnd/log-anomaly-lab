# Code Implementation

## 📁 Cấu Trúc Thư Mục

```
code/
├── preprocessing/          # Log parsing và preprocessing
│   ├── log_parser.py      # Main parser sử dụng Drain3
│   ├── test_parser.py     # Test script (sample nhỏ)
│   └── parse_full_dataset.py  # Parse full dataset (Colab)
├── models/                 # Model implementations
│   ├── autoencoder.py
│   └── logbert.py
├── training/               # Training scripts
│   ├── train_autoencoder.py
│   ├── train_logbert.py
│   └── evaluate.py
├── requirements.txt        # Python dependencies
└── README.md              # File này
```

---

## 🚀 Setup

### 1. Cài Đặt Dependencies

```bash
# Tạo virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# hoặc venv\Scripts\activate  # Windows

# Cài đặt packages
pip install -r requirements.txt
```

### 2. Kiểm Tra Dependencies

```bash
python3 -c "import drain3; print('drain3 OK')"
python3 -c "import torch; print('PyTorch OK')"
python3 -c "import transformers; print('Transformers OK')"
```

---

## 📝 Log Parser

### Test với Sample Nhỏ (Local)

```bash
cd preprocessing
python3 test_parser.py
```

**Output:**
- `hdfs_parsed_sample.json` - HDFS parsed logs (2000 dòng)
- `bgl_parsed_sample.json` - BGL parsed logs (2000 dòng)
- Statistics in console

### Parse Full Dataset (Colab)

```bash
# Upload code lên Colab
# Chạy:
python3 preprocessing/parse_full_dataset.py
```

**Output:**
- `preprocessing/output/hdfs_parsed_full.json`
- `preprocessing/output/bgl_parsed_full.json`
- `preprocessing/output/hdfs_parsing_stats.json`
- `preprocessing/output/bgl_parsing_stats.json`

---

## 📊 Usage

### Sử dụng LogParser trong code

```python
from preprocessing.log_parser import LogParser

# Khởi tạo parser
parser = LogParser()

# Parse một log entry
log_line = "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block..."
result = parser.parse_log(log_line, dataset_type="HDFS")

if result:
    print(f"Template: {result['template']}")
    print(f"Parameters: {result['parameters']}")

# Parse dataset
results = parser.parse_dataset(
    "datasets/HDFS.log",
    dataset_type="HDFS",
    max_lines=2000  # None = parse tất cả
)

# Xem statistics
stats = parser.get_statistics()
print(f"Success rate: {stats['success_rate']:.2f}%")
```

---

## ✅ Checklist

- [x] Setup môi trường
- [x] Tải dataset (HDFS, BGL)
- [ ] Test parser với sample nhỏ
- [ ] Parse full dataset trên Colab
- [ ] Implement tokenization
- [ ] Implement embedding
- [ ] Implement Autoencoder
- [ ] Implement LogBERT
- [ ] Training scripts
- [ ] Evaluation scripts

---

## 🔗 Liên Kết

- [Drain3 Documentation](https://github.com/IBM/Drain3)
- [LogHub Datasets](https://github.com/logpai/loghub)

