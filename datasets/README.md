# Datasets

## Hướng Dẫn Tải Dataset

### HDFS Dataset
- **Nguồn**: [GitHub logpai/loghub](https://github.com/logpai/loghub)
- **Link**: https://github.com/logpai/loghub/tree/master/HDFS
- **Mô tả**: Hadoop Distributed File System logs
- **Kích thước**: ~11M log entries
- **Anomaly ratio**: ~5.2%

**Cách tải:**
```bash
git clone https://github.com/logpai/loghub.git
cd loghub/HDFS
```

---

### BGL Dataset
- **Nguồn**: [GitHub logpai/loghub](https://github.com/logpai/loghub)
- **Link**: https://github.com/logpai/loghub/tree/master/BGL
- **Mô tả**: BlueGene/L supercomputer logs
- **Kích thước**: ~4.7M log entries
- **Anomaly ratio**: ~7.4%

**Cách tải:**
```bash
cd loghub/BGL
```

---

### Dataset Format

Các dataset thường có format:
- **Raw logs**: File text chứa log entries
- **Labels**: File chứa ground truth (normal vs anomaly)
- **Parsed logs** (optional): Log đã được parse thành template

---

## Lưu Ý

- Các dataset có thể lớn, cần đủ disk space
- Dataset thường được chia train/test split
- Cần đọc documentation để hiểu format cụ thể

