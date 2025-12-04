# Hướng Dẫn Chạy Toàn Bộ Pipeline

## Tổng Quan

Tài liệu này hướng dẫn chi tiết cách chạy toàn bộ pipeline từ preprocessing đến evaluation và cập nhật báo cáo.

**Hỗ trợ**: Windows, Linux, macOS

## GIAI ĐOẠN 1: Chuẩn Bị Môi Trường

### Bước 1.1: Kiểm tra môi trường

**Windows (PowerShell hoặc CMD):**
```powershell
cd code
.\venv\Scripts\Activate.ps1    # PowerShell
# hoặc
.\venv\Scripts\activate.bat    # CMD
python --version  # Cần >= 3.8
```

**Linux/Mac:**
```bash
cd code
source venv/bin/activate
python3 --version  # Cần >= 3.8
```

### Bước 1.2: Cài đặt dependencies

**Windows/Linux/Mac:**
```bash
pip install -r requirements.txt
```

### Bước 1.3: Kiểm tra GPU

**Windows/Linux/Mac:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

Nếu chưa có PyTorch với CUDA, cài đặt:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Lưu ý Windows**: Nếu gặp lỗi khi import torch, có thể cần cài đặt Visual C++ Redistributable.

### Bước 1.4: Kiểm tra datasets

**Windows (PowerShell):**
```powershell
Get-ChildItem datasets\HDFS_v1\HDFS.log    # ~1.5GB
Get-ChildItem datasets\BGL.log              # ~709MB
Get-ChildItem datasets\HDFS_v1\preprocessed\anomaly_label.csv
```

**Windows (CMD):**
```cmd
dir datasets\HDFS_v1\HDFS.log
dir datasets\BGL.log
dir datasets\HDFS_v1\preprocessed\anomaly_label.csv
```

**Linux/Mac:**
```bash
ls -lh datasets/HDFS_v1/HDFS.log  # ~1.5GB
ls -lh datasets/BGL.log            # ~709MB
ls -lh datasets/HDFS_v1/preprocessed/anomaly_label.csv
```

## GIAI ĐOẠN 2: Preprocessing

### Cách 1: Chạy từng bước (khuyến nghị cho lần đầu)

**Windows/Linux/Mac:**
```bash
cd preprocessing

# Parse HDFS
python scripts/test_preprocessing.py --parse_full --dataset HDFS

# Parse BGL
python scripts/test_preprocessing.py --parse_full --dataset BGL

# Tokenization & Embedding cho HDFS (Word2Vec)
python scripts/process_full_dataset.py --dataset HDFS --embedding_method word2vec

# Tokenization & Embedding cho HDFS (BERT)
python scripts/process_full_dataset.py --dataset HDFS --embedding_method bert

# Tokenization & Embedding cho BGL (Word2Vec)
python scripts/process_full_dataset.py --dataset BGL --embedding_method word2vec

# Tokenization & Embedding cho BGL (BERT)
python scripts/process_full_dataset.py --dataset BGL --embedding_method bert
```

### Cách 2: Chạy script tự động

**Windows (PowerShell):**
```powershell
cd code
.\run_full_pipeline.ps1
```

**Windows (CMD):**
```cmd
cd code
run_full_pipeline.bat
```

**Linux/Mac:**
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

**Windows/Linux/Mac:**
```bash
cd training

# HDFS
python scripts/train.py --model autoencoder --dataset HDFS --epochs 50 --batch_size 32 --device cuda

# BGL
python scripts/train.py --model autoencoder --dataset BGL --epochs 50 --batch_size 32 --device cuda
```

### Train LogBERT

**Windows/Linux/Mac:**
```bash
# HDFS
python scripts/train.py --model logbert --dataset HDFS --epochs 10 --batch_size 16 --device cuda

# BGL
python scripts/train.py --model logbert --dataset BGL --epochs 10 --batch_size 16 --device cuda
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

**Windows/Linux/Mac:**
```bash
cd training

# Autoencoder
python scripts/tune.py --model autoencoder --dataset HDFS --epochs 10
python scripts/tune.py --model autoencoder --dataset BGL --epochs 10

# LogBERT
python scripts/tune.py --model logbert --dataset HDFS --epochs 5
python scripts/tune.py --model logbert --dataset BGL --epochs 5

# Visualize
python scripts/plot_tuning.py --model autoencoder --dataset both
python scripts/plot_tuning.py --model logbert --dataset both
```

**Thời gian dự kiến**: 12-24 giờ

**Output**:
- `training/output/tuning_results/autoencoder/hdfs_tuning_results.json`
- `training/output/tuning_results/logbert/hdfs_tuning_results.json`
- `training/output/tuning_results/*/tuning_visualization.png`

## GIAI ĐOẠN 5: Threshold Selection

**Windows/Linux/Mac:**
```bash
cd training

python scripts/threshold.py --model_type autoencoder --dataset HDFS
python scripts/threshold.py --model_type autoencoder --dataset BGL
python scripts/threshold.py --model_type logbert --dataset HDFS
python scripts/threshold.py --model_type logbert --dataset BGL
```

**Thời gian dự kiến**: 1-2 giờ

**Output**:
- `training/output/thresholds/autoencoder_hdfs_threshold.json`
- `training/output/thresholds/logbert_hdfs_threshold.json`
- (và tương tự cho BGL)

## GIAI ĐOẠN 6: Evaluation

**Windows/Linux/Mac:**
```bash
cd evaluation

# Evaluate all models
python scripts/evaluate.py --model_type autoencoder --dataset HDFS
python scripts/evaluate.py --model_type autoencoder --dataset BGL
python scripts/evaluate.py --model_type logbert --dataset HDFS
python scripts/evaluate.py --model_type logbert --dataset BGL

# Generate all plots
python scripts/plot.py --plot_type all --model_type both --dataset both
```

**Thời gian dự kiến**: 2-4 giờ

**Output**:
- `evaluation/output/evaluation/autoencoder_hdfs_evaluation.json`
- `evaluation/output/evaluation/logbert_hdfs_evaluation.json`
- ROC curves, confusion matrices, loss curves

## GIAI ĐOẠN 7: Thu Thập Kết Quả

**Windows/Linux/Mac:**
```bash
cd code
python collect_results.py
```

**Output**:
- `results_summary/full_results.json`: Tất cả kết quả
- `results_summary/summary_table.csv`: Bảng tổng hợp (CSV)
- `results_summary/summary_table.md`: Bảng tổng hợp (Markdown)
- `results_summary/report.md`: Báo cáo văn bản

## GIAI ĐOẠN 8: Cập Nhật Báo Cáo

**Windows/Linux/Mac:**
```bash
cd code
python update_chapter2.py
```

**Output**: `chapter2_updates.md` - Nội dung đã format sẵn để copy vào Chương 2

## Chạy Toàn Bộ Pipeline Tự Động

### Chạy tất cả

**Windows (PowerShell):**
```powershell
cd code
.\run_full_pipeline.ps1
```

**Windows (CMD):**
```cmd
cd code
run_full_pipeline.bat
```

**Linux/Mac:**
```bash
cd code
./run_full_pipeline.sh
```

### Chạy từng phần

**Windows (PowerShell):**
```powershell
# Chỉ preprocessing
.\run_full_pipeline.ps1 -SkipTraining -SkipTuning -SkipEvaluation

# Chỉ training (đã có preprocessing)
.\run_full_pipeline.ps1 -SkipPreprocessing -SkipTuning -SkipEvaluation

# Chỉ tuning (đã có training)
.\run_full_pipeline.ps1 -SkipPreprocessing -SkipTraining -SkipEvaluation
```

**Linux/Mac:**
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

**Windows/Linux/Mac:**
```bash
nvidia-smi
```

**Linux/Mac (auto-refresh):**
```bash
watch -n 1 nvidia-smi
```

**Windows (PowerShell - auto-refresh):**
```powershell
while ($true) { Clear-Host; nvidia-smi; Start-Sleep -Seconds 1 }
```

### Monitor Disk Space

**Windows (PowerShell):**
```powershell
Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{Name="Used(GB)";Expression={[math]::Round($_.Used/1GB,2)}}, @{Name="Free(GB)";Expression={[math]::Round($_.Free/1GB,2)}}
```

**Windows (CMD):**
```cmd
wmic logicaldisk get name,freespace,size
```

**Linux/Mac:**
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

**Windows/Linux/Mac:**
```bash
python scripts/train.py --model autoencoder --dataset HDFS --resume path/to/checkpoint.pt
```

**Lưu ý Windows**: Đường dẫn có thể dùng backslash hoặc forward slash:
```bash
python scripts/train.py --model autoencoder --dataset HDFS --resume "training\output\checkpoints\autoencoder_hdfs\best_model.pt"
```

### File Not Found

Kiểm tra paths trong scripts. Đảm bảo đã chạy các bước trước đó.

## Backup Kết Quả

**Windows (PowerShell):**
```powershell
# Backup checkpoints
Compress-Archive -Path training\output\checkpoints\ -DestinationPath checkpoints_backup.zip

# Backup evaluation results
Compress-Archive -Path evaluation\output\ -DestinationPath evaluation_backup.zip

# Backup all results
Compress-Archive -Path results_summary\ -DestinationPath results_backup.zip
```

**Linux/Mac:**
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

---

## Lưu Ý Đặc Biệt Cho Windows

### 1. Virtual Environment

**PowerShell:**
- Nếu gặp lỗi "execution of scripts is disabled", chạy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**CMD:**
- Sử dụng `activate.bat` thay vì `Activate.ps1`

### 2. Đường Dẫn

- Windows hỗ trợ cả forward slash (`/`) và backslash (`\`) trong Python
- Khi dùng trong bash scripts, dùng forward slash
- Khi dùng trong PowerShell/CMD, có thể dùng cả hai

### 3. Python Command

- Trên Windows, thường dùng `python` thay vì `python3`
- Nếu có cả Python 2 và 3, dùng `py -3` hoặc `python3` nếu đã cài đặt

### 4. GPU Support

- Đảm bảo đã cài NVIDIA drivers
- Cài đặt CUDA toolkit từ NVIDIA website
- Cài PyTorch với CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 5. Long Path Support (Windows 10+)

Nếu gặp lỗi "path too long", enable long path support:
```powershell
# Chạy PowerShell as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

### 6. Scripts Tự Động

Nếu không có `run_full_pipeline.ps1` hoặc `run_full_pipeline.bat`, có thể chạy từng bước thủ công theo hướng dẫn ở trên.

