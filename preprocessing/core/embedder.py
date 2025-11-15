"""
Embedding module - Chuyển đổi tokens thành vector representations
Hỗ trợ nhiều phương pháp: TF-IDF, Word2Vec, FastText, BERT
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Union
import logging

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. TF-IDF will not be available.")

try:
    from gensim.models import Word2Vec, FastText
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Warning: gensim not installed. Word2Vec/FastText will not be available.")

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch/transformers not installed. BERT embeddings will not be available.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogEmbedder:
    """
    Embedder cho log messages
    Hỗ trợ nhiều phương pháp embedding
    """
    
    def __init__(self, method: str = "word2vec", embedding_dim: int = 128,
                 bert_model: str = "bert-base-uncased", device: str = "cpu"):
        """
        Khởi tạo embedder
        
        Args:
            method: "tfidf", "word2vec", "fasttext", hoặc "bert"
            embedding_dim: Kích thước embedding vector (cho word2vec/fasttext)
            bert_model: Tên BERT model (cho BERT embeddings)
            device: "cpu" hoặc "cuda" (cho BERT)
        """
        self.method = method
        self.embedding_dim = embedding_dim
        self.bert_model_name = bert_model
        self.device = device
        
        self.model = None
        self.tokenizer = None
        self.is_fitted = False
        
        # BERT setup
        if method == "bert":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch and transformers required for BERT embeddings")
            self._setup_bert()
    
    def _setup_bert(self):
        """Setup BERT model và tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
            self.model = AutoModel.from_pretrained(self.bert_model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded BERT model: {self.bert_model_name} on {self.device}")
            self.is_fitted = True
        except Exception as e:
            logger.error(f"Error loading BERT model: {str(e)}")
            raise
    
    def fit(self, texts: List[str], tokenized_texts: Optional[List[List[str]]] = None):
        """
        Train embedding model (nếu cần)
        
        Args:
            texts: List các log messages hoặc templates
            tokenized_texts: List các tokenized texts (cho word2vec/fasttext)
        """
        if self.method == "tfidf":
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn required for TF-IDF")
            
            self.model = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            self.model.fit(texts)
            self.is_fitted = True
            logger.info(f"Fitted TF-IDF vectorizer on {len(texts)} texts")
        
        elif self.method == "word2vec":
            if not GENSIM_AVAILABLE:
                raise ImportError("gensim required for Word2Vec")
            
            if tokenized_texts is None:
                # Tokenize texts nếu chưa tokenized
                tokenized_texts = [text.split() for text in texts]
            
            self.model = Word2Vec(
                sentences=tokenized_texts,
                vector_size=self.embedding_dim,
                window=5,
                min_count=2,
                workers=4,
                sg=0  # CBOW
            )
            self.is_fitted = True
            logger.info(f"Trained Word2Vec model on {len(tokenized_texts)} texts")
            logger.info(f"Vocabulary size: {len(self.model.wv.key_to_index)}")
        
        elif self.method == "fasttext":
            if not GENSIM_AVAILABLE:
                raise ImportError("gensim required for FastText")
            
            if tokenized_texts is None:
                tokenized_texts = [text.split() for text in texts]
            
            self.model = FastText(
                sentences=tokenized_texts,
                vector_size=self.embedding_dim,
                window=5,
                min_count=2,
                workers=4,
                sg=0
            )
            self.is_fitted = True
            logger.info(f"Trained FastText model on {len(tokenized_texts)} texts")
            logger.info(f"Vocabulary size: {len(self.model.wv.key_to_index)}")
        
        elif self.method == "bert":
            # BERT không cần training, đã pre-trained
            logger.info("BERT model ready (pre-trained, no training needed)")
    
    def embed(self, text: Union[str, List[str]], tokenized: Optional[List[List[str]]] = None) -> np.ndarray:
        """
        Embed một text hoặc list texts
        
        Args:
            text: Text hoặc list texts cần embed
            tokenized: Tokenized texts (cho word2vec/fasttext)
        
        Returns:
            Embedding vector(s) - shape (embedding_dim,) hoặc (n_texts, embedding_dim)
        """
        if isinstance(text, str):
            return self.embed_single(text, tokenized)
        else:
            return self.embed_batch(text, tokenized)
    
    def embed_single(self, text: str, tokenized: Optional[List[str]] = None) -> np.ndarray:
        """
        Embed một text đơn
        
        Args:
            text: Text cần embed
            tokenized: Tokenized text (cho word2vec/fasttext)
        
        Returns:
            Embedding vector - shape (embedding_dim,)
        """
        if not self.is_fitted:
            raise ValueError("Embedder chưa được fit. Gọi fit() trước.")
        
        if self.method == "tfidf":
            # TF-IDF returns sparse matrix, convert to dense
            embedding = self.model.transform([text]).toarray()[0]
            return embedding
        
        elif self.method == "word2vec":
            if tokenized is None:
                tokens = text.split()
            else:
                tokens = tokenized
            
            # Average word embeddings
            embeddings = []
            for token in tokens:
                if token in self.model.wv:
                    embeddings.append(self.model.wv[token])
            
            if len(embeddings) == 0:
                # Return zero vector nếu không có token nào trong vocab
                return np.zeros(self.embedding_dim)
            
            return np.mean(embeddings, axis=0)
        
        elif self.method == "fasttext":
            if tokenized is None:
                tokens = text.split()
            else:
                tokens = tokenized
            
            # Average word embeddings (FastText có thể handle OOV)
            embeddings = [self.model.wv[token] for token in tokens]
            
            if len(embeddings) == 0:
                return np.zeros(self.embedding_dim)
            
            return np.mean(embeddings, axis=0)
        
        elif self.method == "bert":
            # BERT embedding
            encoded = self.tokenizer(
                text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            return embedding
        
        else:
            raise ValueError(f"Unknown embedding method: {self.method}")
    
    def embed_batch(self, texts: List[str], tokenized: Optional[List[List[str]]] = None) -> np.ndarray:
        """
        Embed một batch texts
        
        Args:
            texts: List texts
            tokenized: List tokenized texts
        
        Returns:
            Embedding matrix - shape (n_texts, embedding_dim)
        """
        if self.method == "bert":
            # Batch processing cho BERT (hiệu quả hơn)
            encoded = self.tokenizer(
                texts,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embeddings
        
        else:
            # Process từng text
            embeddings = []
            for i, text in enumerate(texts):
                tok = tokenized[i] if tokenized else None
                emb = self.embed_single(text, tok)
                embeddings.append(emb)
            
            return np.array(embeddings)
    
    def save(self, filepath: str):
        """
        Lưu embedding model
        """
        if self.method == "tfidf":
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Saved TF-IDF model to {filepath}")
        
        elif self.method in ["word2vec", "fasttext"]:
            self.model.save(filepath)
            logger.info(f"Saved {self.method} model to {filepath}")
        
        elif self.method == "bert":
            # BERT model đã được lưu local khi load
            # Chỉ cần lưu config
            config = {
                'method': self.method,
                'bert_model': self.bert_model_name,
                'device': self.device
            }
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved BERT config to {filepath}")
    
    def load(self, filepath: str):
        """
        Load embedding model
        """
        if self.method == "tfidf":
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.is_fitted = True
            logger.info(f"Loaded TF-IDF model from {filepath}")
        
        elif self.method in ["word2vec", "fasttext"]:
            if self.method == "word2vec":
                self.model = Word2Vec.load(filepath)
            else:
                self.model = FastText.load(filepath)
            self.is_fitted = True
            logger.info(f"Loaded {self.method} model from {filepath}")
        
        elif self.method == "bert":
            # BERT sẽ load khi khởi tạo
            logger.info("BERT model loaded during initialization")



