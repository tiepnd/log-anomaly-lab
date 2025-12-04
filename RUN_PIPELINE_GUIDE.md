# Hướng Dẫn Chạy Toàn Bộ Pipeline

## Tổng Quan

Tài liệu này hướng dẫn chi tiết cách chạy toàn bộ pipeline từ preprocessing đến evaluation và cập nhật báo cáo.

## GIAI ĐOẠN 1: Chuẩn Bị Môi Trường

### Bước 1.1: Kiểm tra môi trường

```bash
cd code
source venv/bin/activate
python3 --version  # Cần >= 3.8
```

### Bước 1.2: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 1.3: Kiểm tra GPU

```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

Nếu chưa có PyTorch với CUDA, cài đặt:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Bước 1.4: Kiểm tra datasets

```bash
ls -lh datasets/HDFS_v1/HDFS.log  # ~1.5GB
ls -lh datasets/BGL.log            # ~709MB
ls -lh datasets/HDFS_v1/preprocessed/anomaly_label.csv
```

## GIAI ĐOẠN 2: Preprocessing

### Cách 1: Chạy từng bước (khuyến nghị cho lần đầu)

```bash
cd preprocessing

# Parse HDFS
python3 scripts/test_preprocessing.py --parse_full --dataset HDFS

# Parse BGL
python3 scripts/test_preprocessing.py --parse_full --dataset BGL

# Tokenization & Embedding cho HDFS
python3 scripts/test_preprocessing.py --section tokenization_embedding --dataset HDFS --parse_full

# Tokenization & Embedding cho BGL
python3 scripts/test_preprocessing.py --section tokenization_embedding --dataset BGL --parse_full
```

### Cách 2: Chạy script tự động

```bash
cd code
./run_full_pipeline.sh
```

**Thời gian dự kiến**: 2-4 giờ

**Output**:
- `preprocessing/output/parsed/hdfs_parsed.json`
- `preprocessing/output/parsed/bgl_parsed.json`
- `preprocessing/output/processed/hdfs_processed_word2vec.json`
- `preprocessing/output/processed/hdfs_processed_bert.json`
- `preprocessing/output/processed/bgl_processed_word2vec.json`
- `preprocessing/output/processed/bgl_processed_bert.json`

## GIAI ĐOẠN 3: Training

### Train Autoencoder

```bash
cd training

# HDFS
python3 scripts/train.py --model autoencoder --dataset HDFS --epochs 50 --batch_size 32 --device cuda

# BGL
python3 scripts/train.py --model autoencoder --dataset BGL --epochs 50 --batch_size 32 --device cuda
```

### Train LogBERT

```bash
# HDFS
python3 scripts/train.py --model logbert --dataset HDFS --epochs 10 --batch_size 16 --device cuda

# BGL
python3 scripts/train.py --model logbert --dataset BGL --epochs 10 --batch_size 16 --device cuda
```

**Lưu ý**: Nếu gặp Out of Memory, giảm batch_size:
- Autoencoder: `--batch_size 16` hoặc `--batch_size 8`
- LogBERT: `--batch_size 8` hoặc `--batch_size 4`

**Thời gian dự kiến**: 8-16 giờ

**Output**:
- `training/output/checkpoints/autoencoder_hdfs/best_model.pt`
- `training/output/checkpoints/autoencoder_bgl/best_model.pt`
- `training/output/checkpoints/logbert_hdfs/best_model.pt`
- `training/output/checkpoints/logbert_bgl/best_model.pt`

## GIAI ĐOẠN 4: Hyperparameter Tuning

```bash
cd training

# Autoencoder
python3 scripts/tune.py --model autoencoder --dataset HDFS --epochs 10
python3 scripts/tune.py --model autoencoder --dataset BGL --epochs 10

# LogBERT
python3 scripts/tune.py --model logbert --dataset HDFS --epochs 5
python3 scripts/tune.py --model logbert --dataset BGL --epochs 5

# Visualize
python3 scripts/plot_tuning.py --model autoencoder --dataset both
python3 scripts/plot_tuning.py --model logbert --dataset both
```

**Thời gian dự kiến**: 12-24 giờ

**Output**:
- `training/output/tuning_results/autoencoder/hdfs_tuning_results.json`
- `training/output/tuning_results/logbert/hdfs_tuning_results.json`
- `training/output/tuning_results/*/tuning_visualization.png`

## GIAI ĐOẠN 5: Threshold Selection

```bash
cd training

python3 scripts/threshold.py --model_type autoencoder --dataset HDFS
python3 scripts/threshold.py --model_type autoencoder --dataset BGL
python3 scripts/threshold.py --model_type logbert --dataset HDFS
python3 scripts/threshold.py --model_type logbert --dataset BGL
```

**Thời gian dự kiến**: 1-2 giờ

**Output**:
- `training/output/thresholds/autoencoder_hdfs_threshold.json`
- `training/output/thresholds/logbert_hdfs_threshold.json`
- (và tương tự cho BGL)

## GIAI ĐOẠN 6: Evaluation

```bash
cd evaluation

# Evaluate all models
python3 scripts/evaluate.py --model_type autoencoder --dataset HDFS
python3 scripts/evaluate.py --model_type autoencoder --dataset BGL
python3 scripts/evaluate.py --model_type logbert --dataset HDFS
python3 scripts/evaluate.py --model_type logbert --dataset BGL

# Generate all plots
python3 scripts/plot.py --plot_type all --model_type both --dataset both
```

**Thời gian dự kiến**: 2-4 giờ

**Output**:
- `evaluation/output/evaluation/autoencoder_hdfs_evaluation.json`
- `evaluation/output/evaluation/logbert_hdfs_evaluation.json`
- ROC curves, confusion matrices, loss curves

## GIAI ĐOẠN 7: Thu Thập Kết Quả

```bash
cd code
python3 collect_results.py
```

**Output**:
- `results_summary/full_results.json`: Tất cả kết quả
- `results_summary/summary_table.csv`: Bảng tổng hợp (CSV)
- `results_summary/summary_table.md`: Bảng tổng hợp (Markdown)
- `results_summary/report.md`: Báo cáo văn bản

## GIAI ĐOẠN 8: Cập Nhật Báo Cáo

```bash
cd code
python3 update_chapter2.py
```

**Output**: `chapter2_updates.md` - Nội dung đã format sẵn để copy vào Chương 2

## Chạy Toàn Bộ Pipeline Tự Động

### Chạy tất cả

```bash
cd code
./run_full_pipeline.sh
```

### Chạy từng phần

```bash
# Chỉ preprocessing
./run_full_pipeline.sh --skip-training --skip-tuning --skip-evaluation

# Chỉ training (đã có preprocessing)
./run_full_pipeline.sh --skip-preprocessing --skip-tuning --skip-evaluation

# Chỉ tuning (đã có training)
./run_full_pipeline.sh --skip-preprocessing --skip-training --skip-evaluation
```

## Monitor Quá Trình

### Monitor GPU

```bash
watch -n 1 nvidia-smi
```

### Monitor Disk Space

```bash
df -h
```

### Check Logs

Tất cả scripts đều có logging. Check console output hoặc log files.

## Troubleshooting

### Out of Memory

1. Giảm batch_size
2. Dùng gradient accumulation
3. Clear cache: `torch.cuda.empty_cache()`

### Training bị gián đoạn

Resume từ checkpoint:
```bash
python3 scripts/train.py --model autoencoder --dataset HDFS --resume path/to/checkpoint.pt
```

### File Not Found

Kiểm tra paths trong scripts. Đảm bảo đã chạy các bước trước đó.

## Backup Kết Quả

```bash
# Backup checkpoints
tar -czf checkpoints_backup.tar.gz training/output/checkpoints/

# Backup evaluation results
tar -czf evaluation_backup.tar.gz evaluation/output/

# Backup all results
tar -czf results_backup.tar.gz results_summary/
```

## Timeline Ước Tính

- **Preprocessing**: 2-4 giờ
- **Training**: 8-16 giờ
- **Tuning**: 12-24 giờ
- **Threshold Selection**: 1-2 giờ
- **Evaluation**: 2-4 giờ
- **Tổng cộng**: 25-50 giờ

*Lưu ý: Có thể chạy song song một số bước (ví dụ: train HDFS và BGL song song nếu có nhiều GPU)*

