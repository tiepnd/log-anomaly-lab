"""
LogBERT Model cho Log Anomaly Detection
Dựa trên BERT architecture với fine-tuning cho anomaly detection task
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import logging

try:
    from transformers import (
        AutoModel,
        AutoTokenizer,
        BertForSequenceClassification,
        BertConfig,
        BertModel
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. LogBERT will not be available.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogBERT(nn.Module):
    """
    LogBERT model cho log anomaly detection
    Sử dụng BERT architecture với classification hoặc reconstruction task
    """
    
    def __init__(self,
                 bert_model_name: str = "bert-base-uncased",
                 task: str = "classification",
                 num_labels: int = 2,
                 use_pretrained: bool = True,
                 dropout: float = 0.1,
                 max_length: int = 512):
        """
        Khởi tạo LogBERT model
        
        Args:
            bert_model_name: Tên BERT model ("bert-base-uncased", "distilbert-base-uncased")
            task: "classification" hoặc "reconstruction"
            num_labels: Số labels cho classification (2 = binary)
            use_pretrained: Có sử dụng pre-trained weights không
            dropout: Dropout rate
            max_length: Max sequence length
        """
        super(LogBERT, self).__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for LogBERT")
        
        self.bert_model_name = bert_model_name
        self.task = task
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Load BERT model
        if use_pretrained:
            if task == "classification":
                # Sử dụng BertForSequenceClassification cho classification
                self.bert_model = AutoModel.from_pretrained(bert_model_name)
            else:
                # Sử dụng BertModel cho reconstruction
                self.bert_model = AutoModel.from_pretrained(bert_model_name)
        else:
            # Tạo model từ config (không load weights)
            config = AutoModel.from_pretrained(bert_model_name).config
            self.bert_model = AutoModel.from_config(config)
        
        # Get hidden size
        self.hidden_size = self.bert_model.config.hidden_size
        
        # Classification head (cho classification task)
        if task == "classification":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size // 2, num_labels)
            )
        else:
            # Reconstruction head (cho reconstruction task)
            self.reconstruction_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, self.hidden_size)
            )
        
        logger.info(f"Initialized LogBERT:")
        logger.info(f"  BERT model: {bert_model_name}")
        logger.info(f"  Task: {task}")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Max length: {max_length}")
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Labels cho training (batch_size,) hoặc (batch_size, num_labels)
        
        Returns:
            Dict với keys: 'logits', 'loss' (nếu có labels), 'hidden_states'
        """
        # BERT forward
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get [CLS] token representation (first token)
        cls_hidden = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        if self.task == "classification":
            # Classification task
            logits = self.classifier(cls_hidden)
            
            outputs_dict = {
                'logits': logits,
                'hidden_states': cls_hidden
            }
            
            # Calculate loss nếu có labels
            if labels is not None:
                if labels.dim() == 1:
                    # Binary classification
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)
                else:
                    # Multi-label classification
                    loss_fn = nn.BCEWithLogitsLoss()
                    loss = loss_fn(logits, labels.float())
                
                outputs_dict['loss'] = loss
            
            return outputs_dict
        
        else:
            # Reconstruction task
            reconstructed = self.reconstruction_head(cls_hidden)
            
            outputs_dict = {
                'reconstructed': reconstructed,
                'hidden_states': cls_hidden,
                'original': cls_hidden  # Original embedding
            }
            
            # Calculate reconstruction loss nếu có labels (original embeddings)
            if labels is not None:
                loss_fn = nn.MSELoss()
                loss = loss_fn(reconstructed, labels)
                outputs_dict['loss'] = loss
            
            return outputs_dict
    
    def get_anomaly_score(self,
                         input_ids: torch.Tensor,
                         attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Tính anomaly score
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            Anomaly scores (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            if self.task == "classification":
                # Softmax probabilities
                probs = F.softmax(outputs['logits'], dim=1)
                # Anomaly score = probability of anomaly class (class 1)
                anomaly_scores = probs[:, 1] if self.num_labels == 2 else probs[:, 1:].sum(dim=1)
            else:
                # Reconstruction error
                reconstructed = outputs['reconstructed']
                original = outputs['original']
                error = F.mse_loss(reconstructed, original, reduction='none')
                anomaly_scores = error.mean(dim=1)
            
            return anomaly_scores
    
    def predict_anomaly(self,
                       input_ids: torch.Tensor,
                       attention_mask: Optional[torch.Tensor] = None,
                       threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict anomaly
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            threshold: Threshold cho anomaly detection
        
        Returns:
            Tuple of (predictions, anomaly_scores)
            predictions: Binary predictions (1 = anomaly, 0 = normal)
            anomaly_scores: Continuous anomaly scores
        """
        anomaly_scores = self.get_anomaly_score(input_ids, attention_mask)
        predictions = (anomaly_scores > threshold).long()
        return predictions, anomaly_scores
    
    def get_embeddings(self,
                      input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get embeddings từ [CLS] token
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
        
        Returns:
            Embeddings (batch_size, hidden_size)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            return outputs['hidden_states']


def create_logbert(bert_model_name: str = "bert-base-uncased",
                   task: str = "classification",
                   num_labels: int = 2,
                   use_pretrained: bool = True,
                   **kwargs) -> LogBERT:
    """
    Factory function để tạo LogBERT model
    
    Args:
        bert_model_name: Tên BERT model
        task: "classification" hoặc "reconstruction"
        num_labels: Số labels
        use_pretrained: Có sử dụng pre-trained weights không
        **kwargs: Additional arguments cho LogBERT
    
    Returns:
        LogBERT model
    """
    return LogBERT(
        bert_model_name=bert_model_name,
        task=task,
        num_labels=num_labels,
        use_pretrained=use_pretrained,
        **kwargs
    )


def count_parameters(model: nn.Module) -> int:
    """
    Đếm số lượng parameters trong model
    
    Args:
        model: PyTorch model
    
    Returns:
        Tổng số parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """
    Test LogBERT model
    """
    if not TRANSFORMERS_AVAILABLE:
        print("⚠️  transformers library not available. Cannot test LogBERT.")
        return
    
    print("\n" + "="*70)
    print("TEST LOGBERT MODEL")
    print("="*70)
    
    # Test với sample data
    batch_size = 4
    seq_length = 128
    
    # Create model
    model = create_logbert(
        bert_model_name="bert-base-uncased",
        task="classification",
        num_labels=2,
        use_pretrained=True
    )
    
    print(f"\n✅ Model created:")
    print(f"   Parameters: {count_parameters(model):,}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Sample log messages
    sample_logs = [
        "dfs.DataNode$DataXceiver: Receiving block blk_123",
        "dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job.jar",
        "ERROR: Connection timeout to database server",
        "INFO: Service started successfully"
    ]
    
    # Tokenize
    encoded = tokenizer(
        sample_logs,
        max_length=seq_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    print(f"\n✅ Tokenized:")
    print(f"   Input IDs shape: {encoded['input_ids'].shape}")
    print(f"   Attention mask shape: {encoded['attention_mask'].shape}")
    
    # Forward pass
    outputs = model(
        input_ids=encoded['input_ids'],
        attention_mask=encoded['attention_mask']
    )
    
    print(f"\n✅ Forward pass:")
    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Hidden states shape: {outputs['hidden_states'].shape}")
    
    # Anomaly scores
    anomaly_scores = model.get_anomaly_score(
        encoded['input_ids'],
        encoded['attention_mask']
    )
    
    print(f"\n✅ Anomaly scores:")
    for i, (log, score) in enumerate(zip(sample_logs, anomaly_scores)):
        print(f"   {i+1}. {log[:50]}...")
        print(f"      Score: {score.item():.4f}")
    
    # Predictions
    predictions, scores = model.predict_anomaly(
        encoded['input_ids'],
        encoded['attention_mask'],
        threshold=0.5
    )
    
    print(f"\n✅ Anomaly predictions (threshold=0.5):")
    for i, (log, pred, score) in enumerate(zip(sample_logs, predictions, scores)):
        status = "ANOMALY" if pred.item() == 1 else "NORMAL"
        print(f"   {i+1}. {log[:50]}...")
        print(f"      Status: {status}, Score: {score.item():.4f}")
    
    print("\n" + "="*70)
    print("✅ LogBERT model test completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

