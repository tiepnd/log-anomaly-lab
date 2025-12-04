# Quick Start Guide - Chạy Pipeline Trên Máy Mạnh

## Tóm Tắt

Tài liệu này hướng dẫn nhanh cách chạy toàn bộ pipeline trên máy mạnh với dataset lớn.

## Bước 1: Chuẩn Bị (5 phút)

```bash
cd code
source venv/bin/activate
pip install -r requirements.txt
```

Kiểm tra GPU:
```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## Bước 2: Chạy Pipeline (25-50 giờ)

### Cách 1: Chạy tự động (Khuyến nghị)

**Windows (PowerShell):**
```powershell
.\run_full_pipeline.ps1
```

**Windows (CMD):**
```cmd
run_full_pipeline.bat
```

**Linux/Mac:**
```bash
./run_full_pipeline.sh
```

Script này sẽ tự động chạy tất cả các bước:
1. Preprocessing (2-4 giờ)
2. Training (8-16 giờ)
3. Tuning (12-24 giờ)
4. Threshold Selection (1-2 giờ)
5. Evaluation (2-4 giờ)

### Cách 2: Chạy từng bước (Nếu muốn kiểm soát)

Xem chi tiết trong `RUN_PIPELINE_GUIDE.md`

## Bước 3: Thu Thập Kết Quả (5 phút)

**Windows/Linux/Mac:**
```bash
python collect_results.py
```

Output: `results_summary/` folder với:
- `full_results.json`: Tất cả kết quả
- `summary_table.csv`: Bảng tổng hợp
- `report.md`: Báo cáo văn bản

## Bước 4: Cập Nhật Báo Cáo (10 phút)

**Windows/Linux/Mac:**
```bash
python update_chapter2.py
```

Output: `chapter2_updates.md` - Nội dung đã format sẵn để copy vào Chương 2

## Files Quan Trọng

- `run_full_pipeline.sh` / `run_full_pipeline.ps1` / `run_full_pipeline.bat`: Script chạy toàn bộ pipeline
- `collect_results.py`: Thu thập và tổng hợp kết quả
- `update_chapter2.py`: Tạo nội dung cập nhật Chương 2
- `RUN_PIPELINE_GUIDE.md`: Hướng dẫn chi tiết từng bước

## Lưu Ý

1. **Thời gian**: Pipeline mất 25-50 giờ, có thể chạy qua đêm
2. **GPU**: Cần GPU NVIDIA với 8GB+ VRAM cho LogBERT
3. **Disk Space**: Cần ~10-20GB cho checkpoints và outputs
4. **Monitoring**: 
   - Windows/Linux/Mac: `nvidia-smi`
   - Linux/Mac (auto-refresh): `watch -n 1 nvidia-smi`
   - Windows PowerShell (auto-refresh): `while ($true) { Clear-Host; nvidia-smi; Start-Sleep -Seconds 1 }`

## Troubleshooting

Nếu gặp lỗi Out of Memory:
- Giảm `--batch_size` trong training scripts
- Dùng `--device cpu` nếu không có GPU (chậm hơn nhiều)

Nếu training bị gián đoạn:
- Scripts tự động save checkpoints
- Có thể resume từ checkpoint

## Kết Quả Mong Đợi

Sau khi chạy xong, bạn sẽ có:
- ✅ 4 trained models (Autoencoder + LogBERT, HDFS + BGL)
- ✅ Tuning results với best hyperparameters
- ✅ Evaluation metrics (Precision, Recall, F1, ROC-AUC, etc.)
- ✅ Plots (ROC curves, confusion matrices, loss curves)
- ✅ Formatted content để cập nhật Chương 2

