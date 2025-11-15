"""
Complete Pipeline: Parsing → Tokenization → Embedding
Tích hợp tất cả các bước preprocessing
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

from .parser import LogParser
from .tokenizer import LogTokenizer
from .embedder import LogEmbedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogPreprocessingPipeline:
    """
    Complete preprocessing pipeline cho log anomaly detection
    """
    
    def __init__(self, 
                 parser_persistence_dir: str = "drain3_state",
                 tokenizer_method: str = "word",
                 embedder_method: str = "word2vec",
                 vocab_size: int = 10000,
                 embedding_dim: int = 128,
                 max_length: int = 512,
                 bert_model: str = "bert-base-uncased"):
        """
        Khởi tạo pipeline
        
        Args:
            parser_persistence_dir: Thư mục lưu Drain3 state
            tokenizer_method: "word", "subword", hoặc "bert"
            embedder_method: "tfidf", "word2vec", "fasttext", hoặc "bert"
            vocab_size: Kích thước vocabulary (cho word-level tokenizer)
            embedding_dim: Kích thước embedding vector
            max_length: Độ dài tối đa sequence
            bert_model: Tên BERT model (nếu dùng BERT)
        """
        # Initialize components
        self.parser = LogParser(persistence_dir=parser_persistence_dir)
        
        self.tokenizer = LogTokenizer(
            method=tokenizer_method,
            vocab_size=vocab_size,
            max_length=max_length,
            bert_model=bert_model
        )
        
        self.embedder = LogEmbedder(
            method=embedder_method,
            embedding_dim=embedding_dim,
            bert_model=bert_model
        )
        
        self.is_fitted = False
    
    def fit(self, parsed_logs: List[Dict], dataset_type: str = "HDFS"):
        """
        Fit pipeline trên parsed logs
        
        Args:
            parsed_logs: List các parsed logs từ LogParser
            dataset_type: "HDFS" hoặc "BGL"
        """
        # Extract templates từ parsed logs
        templates = [log['template'] for log in parsed_logs if log.get('template')]
        
        logger.info(f"Fitting pipeline on {len(templates)} templates")
        
        # Fit tokenizer
        if self.tokenizer.method == "word":
            self.tokenizer.fit(templates)
        else:
            # BERT tokenizer không cần fit
            self.tokenizer.fit(templates)
        
        # Fit embedder
        if self.embedder.method in ["word2vec", "fasttext"]:
            # Tokenize cho Word2Vec/FastText
            tokenized_templates = []
            for template in templates:
                if self.tokenizer.method == "word":
                    tokenized = self.tokenizer._word_tokenize(template)
                else:
                    tokenized = template.split()
                tokenized_templates.append(tokenized)
            
            self.embedder.fit(templates, tokenized_templates)
        else:
            # TF-IDF hoặc BERT
            self.embedder.fit(templates)
        
        self.is_fitted = True
        logger.info("Pipeline fitting completed")
    
    def process_log(self, log_line: str, dataset_type: str = "HDFS") -> Dict:
        """
        Process một log line qua toàn bộ pipeline
        
        Args:
            log_line: Raw log line
            dataset_type: "HDFS" hoặc "BGL"
        
        Returns:
            Dict với keys: parsed, tokenized, embedded
        """
        # Step 1: Parse
        parsed = self.parser.parse_log(log_line, dataset_type)
        
        if not parsed:
            return {
                'parsed': None,
                'tokenized': None,
                'embedded': None,
                'error': 'Parsing failed'
            }
        
        template = parsed['template']
        
        # Step 2: Tokenize
        if not self.is_fitted:
            raise ValueError("Pipeline chưa được fit. Gọi fit() trước.")
        
        tokenized = self.tokenizer.tokenize(template)
        
        # Step 3: Embed
        embedded = self.embedder.embed_single(template)
        
        return {
            'parsed': parsed,
            'tokenized': tokenized,
            'embedded': embedded.tolist() if isinstance(embedded, np.ndarray) else embedded,
            'template': template
        }
    
    def process_batch(self, log_lines: List[str], dataset_type: str = "HDFS") -> List[Dict]:
        """
        Process batch log lines
        
        Args:
            log_lines: List raw log lines
            dataset_type: "HDFS" hoặc "BGL"
        
        Returns:
            List các processed results
        """
        results = []
        for log_line in log_lines:
            result = self.process_log(log_line, dataset_type)
            results.append(result)
        return results
    
    def process_parsed_logs(self, parsed_logs: List[Dict]) -> List[Dict]:
        """
        Process từ parsed logs (bỏ qua parsing step)
        
        Args:
            parsed_logs: List các parsed logs
        
        Returns:
            List các processed results
        """
        if not self.is_fitted:
            raise ValueError("Pipeline chưa được fit. Gọi fit() trước.")
        
        results = []
        templates = []
        
        for parsed in parsed_logs:
            if parsed and parsed.get('template'):
                templates.append(parsed['template'])
        
        # Batch tokenize và embed
        if self.embedder.method == "bert":
            # BERT batch processing
            embeddings = self.embedder.embed_batch(templates)
        else:
            # Process từng template
            embeddings = []
            for template in templates:
                emb = self.embedder.embed_single(template)
                embeddings.append(emb)
            embeddings = np.array(embeddings)
        
        # Combine results
        for i, parsed in enumerate(parsed_logs):
            if parsed and parsed.get('template'):
                template = parsed['template']
                tokenized = self.tokenizer.tokenize(template)
                
                result = {
                    'parsed': parsed,
                    'tokenized': tokenized,
                    'embedded': embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i],
                    'template': template
                }
                results.append(result)
            else:
                results.append({
                    'parsed': None,
                    'tokenized': None,
                    'embedded': None,
                    'error': 'Parsing failed'
                })
        
        return results
    
    def save(self, directory: str):
        """
        Lưu toàn bộ pipeline
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save tokenizer
        if self.tokenizer.method == "word":
            self.tokenizer.save(os.path.join(directory, "tokenizer.json"))
        
        # Save embedder
        if self.embedder.method == "tfidf":
            self.embedder.save(os.path.join(directory, "embedder_tfidf.pkl"))
        elif self.embedder.method in ["word2vec", "fasttext"]:
            self.embedder.save(os.path.join(directory, f"embedder_{self.embedder.method}.model"))
        elif self.embedder.method == "bert":
            self.embedder.save(os.path.join(directory, "embedder_bert.json"))
        
        # Save config
        config = {
            'tokenizer_method': self.tokenizer.method,
            'embedder_method': self.embedder.method,
            'vocab_size': self.tokenizer.vocab_size,
            'embedding_dim': self.embedder.embedding_dim,
            'max_length': self.tokenizer.max_length,
            'bert_model': self.embedder.bert_model_name if hasattr(self.embedder, 'bert_model_name') else None
        }
        
        with open(os.path.join(directory, "pipeline_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved pipeline to {directory}")


def process_dataset_pipeline(parsed_logs_file: str, 
                            output_file: str,
                            tokenizer_method: str = "word",
                            embedder_method: str = "word2vec",
                            embedding_dim: int = 128,
                            max_lines: Optional[int] = None):
    """
    Process parsed logs file qua pipeline
    
    Args:
        parsed_logs_file: File JSON chứa parsed logs
        output_file: File output để lưu processed results
        tokenizer_method: Tokenization method
        embedder_method: Embedding method
        embedding_dim: Embedding dimension
        max_lines: Số dòng tối đa để process (None = tất cả)
    """
    # Load parsed logs
    logger.info(f"Loading parsed logs from {parsed_logs_file}")
    with open(parsed_logs_file, 'r') as f:
        parsed_logs = json.load(f)
    
    if max_lines:
        parsed_logs = parsed_logs[:max_lines]
    
    logger.info(f"Processing {len(parsed_logs)} parsed logs")
    
    # Initialize pipeline
    pipeline = LogPreprocessingPipeline(
        tokenizer_method=tokenizer_method,
        embedder_method=embedder_method,
        embedding_dim=embedding_dim
    )
    
    # Fit pipeline
    pipeline.fit(parsed_logs)
    
    # Process logs
    processed_logs = pipeline.process_parsed_logs(parsed_logs)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(processed_logs, f, indent=2)
    
    logger.info(f"Saved processed logs to {output_file}")
    logger.info(f"Processed {len(processed_logs)} logs")
    
    # Statistics
    success_count = sum(1 for r in processed_logs if r.get('embedded') is not None)
    logger.info(f"Success rate: {success_count/len(processed_logs)*100:.2f}%")



