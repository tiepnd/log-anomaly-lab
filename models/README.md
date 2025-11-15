# Models Implementation

## ✅ Autoencoder Model

### Files:
- `autoencoder.py` - Autoencoder model implementation
- `__init__.py` - Module exports

### Architecture:

```
Input Embedding (128 dim)
    ↓
Encoder: [128 → 256 → 128 → 64] → Latent (32 dim)
    ↓
Decoder: [32 → 64 → 128 → 256] → Output (128 dim)
    ↓
Reconstruction Error (MSE)
```

### Usage:

```python
from models.autoencoder import Autoencoder, create_autoencoder

# Create model
model = create_autoencoder(
    input_dim=128,
    hidden_dims=[256, 128, 64],
    latent_dim=32,
    activation="relu",
    dropout=0.1
)

# Forward pass
x = torch.randn(batch_size, 128)
reconstructed, latent = model(x)

# Reconstruction error
error = model.get_reconstruction_error(x)

# Anomaly prediction
threshold = 0.1
predictions = model.predict_anomaly(x, threshold)
```

### Test:

```bash
cd code/models
python3 autoencoder.py
```

---

## 📊 Model Configuration

### Default Hyperparameters:
- **Input dim:** 128 (Word2Vec embedding)
- **Hidden dims:** [256, 128, 64]
- **Latent dim:** 32
- **Activation:** ReLU
- **Dropout:** 0.1

### Training:
- **Learning rate:** 0.001
- **Batch size:** 64
- **Epochs:** 50-100
- **Optimizer:** Adam
- **Loss:** MSE (Mean Squared Error)

---

## ✅ LogBERT Model

### Files:
- `logbert.py` - LogBERT model implementation
- `__init__.py` - Module exports

### Architecture:

```
Tokenized Log Sequence
    ↓
BERT Embedding (Token + Position)
    ↓
Transformer Blocks (Multi-head Attention + FFN)
    ↓
[CLS] Token → Classification/Reconstruction Head
    ↓
Anomaly Score / Binary Classification
```

### Usage:

```python
from models.logbert import LogBERT, create_logbert
from transformers import AutoTokenizer

# Create model
model = create_logbert(
    bert_model_name="bert-base-uncased",
    task="classification",  # hoặc "reconstruction"
    num_labels=2,
    use_pretrained=True
)

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
encoded = tokenizer(log_messages, max_length=128, padding=True, 
                   truncation=True, return_tensors="pt")

# Forward pass
outputs = model(encoded['input_ids'], encoded['attention_mask'])
logits = outputs['logits']

# Anomaly prediction
predictions, anomaly_scores = model.predict_anomaly(
    encoded['input_ids'],
    encoded['attention_mask'],
    threshold=0.5
)
```

### Test:

```bash
cd code/models
python3 logbert.py
```

### Supported BERT Models:

- `bert-base-uncased` (110M parameters)
- `distilbert-base-uncased` (66M parameters) - Nhẹ hơn
- `bert-large-uncased` (340M parameters) - Lớn hơn

### Tasks:

1. **Classification**: Binary classification (normal vs anomaly)
2. **Reconstruction**: Masked Language Modeling (MLM) cho reconstruction loss

---

## 📊 Model Configuration

### Autoencoder Default Hyperparameters:
- **Input dim:** 128 (Word2Vec embedding)
- **Hidden dims:** [256, 128, 64]
- **Latent dim:** 32
- **Activation:** ReLU
- **Dropout:** 0.1

### LogBERT Default Hyperparameters:
- **BERT model:** bert-base-uncased
- **Hidden size:** 768 (bert-base)
- **Task:** classification
- **Num labels:** 2
- **Dropout:** 0.1

### Training:
- **Learning rate:** 0.001 (Autoencoder), 2e-5 (LogBERT)
- **Batch size:** 64 (Autoencoder), 16-32 (LogBERT)
- **Epochs:** 50-100 (Autoencoder), 3-5 (LogBERT fine-tuning)
- **Optimizer:** Adam (Autoencoder), AdamW (LogBERT)
- **Loss:** MSE (Autoencoder), CrossEntropy (LogBERT classification)

---

## 🎯 Next Steps

- [x] Autoencoder model implementation
- [x] LogBERT model implementation
- [ ] Model evaluation scripts
- [ ] Model comparison scripts

