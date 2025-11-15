"""
Dataset classes cho training
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Dataset cho log embeddings"""
    
    def __init__(self, embeddings: np.ndarray):
        """
        Args:
            embeddings: numpy array of embeddings (N, D)
        """
        self.embeddings = torch.FloatTensor(embeddings)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx]


class LogDataset(Dataset):
    """Dataset cho log sequences với BERT tokenization"""
    
    def __init__(self, texts: list, tokenizer, max_length: int = 128, labels: list = None):
        """
        Args:
            texts: List log messages hoặc templates
            tokenizer: BERT tokenizer
            max_length: Max sequence length
            labels: Labels cho training (None = unsupervised)
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

