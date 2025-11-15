"""
Tokenization module - Chuyển đổi parsed logs thành tokens
Hỗ trợ nhiều phương pháp: Word-level, Subword (BERT), Character-level
"""

import os
import json
import re
from typing import List, Dict, Optional, Union
from collections import Counter
import logging

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. BERT tokenization will not be available.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogTokenizer:
    """
    Tokenizer cho log messages
    Hỗ trợ nhiều phương pháp tokenization
    """
    
    def __init__(self, method: str = "word", vocab_size: int = 10000, 
                 max_length: int = 512, bert_model: str = "bert-base-uncased"):
        """
        Khởi tạo tokenizer
        
        Args:
            method: "word", "subword" (BERT), hoặc "char"
            vocab_size: Kích thước vocabulary (cho word-level)
            max_length: Độ dài tối đa của sequence
            bert_model: Tên BERT model (cho subword method)
        """
        self.method = method
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab = Counter()
        self.is_fitted = False
        
        # BERT tokenizer
        if method == "subword" or method == "bert":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers library required for BERT tokenization")
            self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
            logger.info(f"Loaded BERT tokenizer: {bert_model}")
    
    def fit(self, texts: List[str]):
        """
        Build vocabulary từ danh sách texts
        
        Args:
            texts: List các log messages hoặc templates
        """
        if self.method == "word":
            # Build vocabulary từ word-level tokens
            for text in texts:
                tokens = self._word_tokenize(text)
                self.vocab.update(tokens)
            
            # Lấy top vocab_size từ thường gặp nhất
            most_common = self.vocab.most_common(self.vocab_size - 2)  # -2 cho <PAD> và <UNK>
            
            # Tạo word_to_id mapping
            self.word_to_id = {
                "<PAD>": 0,
                "<UNK>": 1,
                "<START>": 2,
                "<END>": 3
            }
            
            for i, (word, count) in enumerate(most_common, start=4):
                self.word_to_id[word] = i
            
            # Tạo id_to_word mapping
            self.id_to_word = {v: k for k, v in self.word_to_id.items()}
            
            self.is_fitted = True
            logger.info(f"Built vocabulary with {len(self.word_to_id)} tokens")
        
        elif self.method in ["subword", "bert"]:
            # BERT tokenizer không cần fit, đã có sẵn vocabulary
            self.is_fitted = True
            logger.info("BERT tokenizer ready (no fitting needed)")
    
    def _word_tokenize(self, text: str) -> List[str]:
        """
        Word-level tokenization
        """
        # Lowercase và split
        text = text.lower()
        # Tách theo whitespace và punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens
    
    def _char_tokenize(self, text: str) -> List[str]:
        """
        Character-level tokenization
        """
        return list(text)
    
    def tokenize(self, text: str, padding: bool = True, 
                 truncation: bool = True) -> Dict[str, Union[List[int], List[str]]]:
        """
        Tokenize một text
        
        Args:
            text: Text cần tokenize
            padding: Có padding đến max_length không
            truncation: Có truncate nếu quá dài không
        
        Returns:
            Dict với keys: 'token_ids', 'tokens', 'attention_mask' (nếu BERT)
        """
        if self.method == "word":
            if not self.is_fitted:
                raise ValueError("Tokenizer chưa được fit. Gọi fit() trước.")
            
            tokens = self._word_tokenize(text)
            token_ids = [self.word_to_id.get(token, self.word_to_id["<UNK>"]) 
                        for token in tokens]
            
            # Padding/Truncation
            if truncation and len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            
            if padding and len(token_ids) < self.max_length:
                token_ids = token_ids + [self.word_to_id["<PAD>"]] * (self.max_length - len(token_ids))
            
            return {
                'token_ids': token_ids,
                'tokens': tokens,
                'length': len(tokens)
            }
        
        elif self.method in ["subword", "bert"]:
            # BERT tokenization
            encoded = self.bert_tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length' if padding else False,
                truncation=truncation,
                return_tensors=None  # Return as lists, not tensors
            )
            
            return {
                'token_ids': encoded['input_ids'],
                'tokens': self.bert_tokenizer.convert_ids_to_tokens(encoded['input_ids']),
                'attention_mask': encoded.get('attention_mask', []),
                'length': len([t for t in encoded['input_ids'] if t != self.bert_tokenizer.pad_token_id])
            }
        
        elif self.method == "char":
            tokens = self._char_tokenize(text)
            # Convert to IDs (simple ASCII-based)
            token_ids = [ord(c) if ord(c) < 128 else 0 for c in tokens]
            
            if truncation and len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            
            if padding and len(token_ids) < self.max_length:
                token_ids = token_ids + [0] * (self.max_length - len(token_ids))
            
            return {
                'token_ids': token_ids,
                'tokens': tokens,
                'length': len(tokens)
            }
        
        else:
            raise ValueError(f"Unknown tokenization method: {self.method}")
    
    def tokenize_batch(self, texts: List[str], padding: bool = True,
                      truncation: bool = True) -> List[Dict]:
        """
        Tokenize một batch texts
        
        Args:
            texts: List các texts
            padding: Có padding không
            truncation: Có truncation không
        
        Returns:
            List các tokenized results
        """
        results = []
        for text in texts:
            result = self.tokenize(text, padding=padding, truncation=truncation)
            results.append(result)
        return results
    
    def save(self, filepath: str):
        """
        Lưu tokenizer (vocabulary và config)
        """
        if self.method == "word":
            data = {
                'method': self.method,
                'vocab_size': self.vocab_size,
                'max_length': self.max_length,
                'word_to_id': self.word_to_id,
                'id_to_word': self.id_to_word
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved tokenizer to {filepath}")
        else:
            logger.warning(f"Save not implemented for method: {self.method}")
    
    def load(self, filepath: str):
        """
        Load tokenizer từ file
        """
        if self.method == "word":
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.method = data['method']
            self.vocab_size = data['vocab_size']
            self.max_length = data['max_length']
            self.word_to_id = data['word_to_id']
            self.id_to_word = {int(k): v for k, v in data['id_to_word'].items()}
            self.is_fitted = True
            logger.info(f"Loaded tokenizer from {filepath}")
        else:
            logger.warning(f"Load not implemented for method: {self.method}")



