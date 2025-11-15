"""
Test script tập trung cho toàn bộ preprocessing pipeline
Tất cả test code được tập trung ở đây

Usage:
    # Test tất cả (với HDFS_2k.log mặc định)
    python3 scripts/test_preprocessing.py

    # Test với file log cụ thể
    python3 scripts/test_preprocessing.py --log_file datasets/HDFS.log

    # Test chỉ parsing
    python3 scripts/test_preprocessing.py --section parsing

    # Parse full dataset (Colab)
    python3 scripts/test_preprocessing.py --parse_full --dataset HDFS
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import argparse
from datetime import datetime

# Add parent directories to path để import preprocessing module
script_dir = Path(__file__).parent.absolute()
preprocessing_dir = script_dir.parent.absolute()
code_dir = preprocessing_dir.parent.absolute()

# Add code/ directory vào path để import preprocessing
sys.path.insert(0, str(code_dir))
# Add preprocessing/ directory vào path
sys.path.insert(0, str(preprocessing_dir))

from preprocessing.core import LogParser, LogTokenizer, LogEmbedder, LogPreprocessingPipeline

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SECTION 1: TEST PARSING
# ============================================================================

def test_parser_single_log():
    """Test parse một log entry đơn lẻ"""
    print("\n" + "="*70)
    print("TEST PARSING SINGLE LOG ENTRIES")
    print("="*70)
    
    parser = LogParser()
    
    # Test HDFS log
    hdfs_log = "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010"
    print("\n📝 HDFS Log:")
    print(f"  Input: {hdfs_log}")
    result = parser.parse_log(hdfs_log, dataset_type="HDFS")
    if result:
        print(f"  ✅ Template: {result['template']}")
        print(f"  ✅ Parameters: {result['parameters']}")
    
    # Test BGL log
    bgl_log = "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
    print("\n📝 BGL Log:")
    print(f"  Input: {bgl_log}")
    result = parser.parse_log(bgl_log, dataset_type="BGL")
    if result:
        print(f"  ✅ Template: {result['template']}")
        print(f"  ✅ Parameters: {result['parameters']}")


def test_parser_sample(log_file: str = None, dataset_type: str = "HDFS", max_lines: int = 2000):
    """
    Test parsing dataset với sample nhỏ
    
    Args:
        log_file: Đường dẫn đến file log (default: datasets/HDFS_2k.log)
        dataset_type: "HDFS" hoặc "BGL"
        max_lines: Số dòng tối đa để parse
    """
    print("\n" + "="*70)
    print(f"TEST PARSING {dataset_type} DATASET (Sample: {max_lines} dòng)")
    print("="*70)
    
    # Tìm file log
    # base_dir là thư mục "code"
    base_dir = Path(__file__).parent.parent.parent
    if log_file is None:
        # Default: HDFS_2k.log cho local test
        log_file = base_dir / "datasets" / "HDFS_2k.log"
        if not log_file.exists():
            # Fallback: HDFS.log
            log_file = base_dir / "datasets" / "HDFS.log"
    else:
        log_file = Path(log_file)
        if not log_file.is_absolute():
            # Nếu relative path, tìm từ base_dir (master's thesis)
            log_file = base_dir / log_file
    
    if not log_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file log: {log_file}")
    
    print(f"📁 Log file: {log_file}")
    
    # Khởi tạo parser
    persistence_dir = str(Path(__file__).parent.parent / "output" / "drain3_state")
    parser = LogParser(persistence_dir=persistence_dir)
    
    # Parse sample
    output_dir = Path(__file__).parent.parent / "output" / "parsed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset_type.lower()}_parsed_sample.json"
    
    results = parser.parse_dataset(
        str(log_file),
        dataset_type=dataset_type,
        max_lines=max_lines,
        output_file=str(output_file)
    )
    
    # In kết quả
    print(f"\n✅ Đã parse thành công {len(results)} log entries")
    
    # In một số ví dụ
    print("\n📋 VÍ DỤ PARSED LOGS (5 ví dụ đầu):")
    print("-" * 70)
    for i, result in enumerate(results[:5], 1):
        print(f"\nVí dụ {i}:")
        print(f"  Template ID: {result['template_id']}")
        print(f"  Template: {result['template']}")
        print(f"  Parameters: {result['parameters']}")
        print(f"  Original: {result['original_message'][:80]}...")
    
    # Statistics
    stats = parser.get_statistics()
    print(f"\n📊 THỐNG KÊ:")
    print(f"  - Success rate: {stats['success_rate']:.2f}%")
    print(f"  - Unique templates: {stats['unique_templates']}")
    print(f"  - Output file: {output_file}")
    
    return parser, results


def parse_full_dataset(dataset_name: str = "HDFS", max_lines: int = None):
    """
    Parse full dataset (chạy trên Colab)
    
    Args:
        dataset_name: "HDFS" hoặc "BGL"
        max_lines: Số dòng tối đa (None = parse tất cả)
    """
    print(f"\n{'='*70}")
    print(f"PARSING FULL {dataset_name} DATASET")
    print(f"{'='*70}\n")
    
    # Setup paths
    # base_dir là thư mục "code"
    base_dir = Path(__file__).parent.parent.parent
    datasets_dir = base_dir / "datasets"
    output_dir = base_dir / "preprocessing" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # File paths
    log_file = datasets_dir / f"{dataset_name}.log"
    output_file = output_dir / "parsed" / f"{dataset_name.lower()}_parsed_full.json"
    stats_file = output_dir / "parsed" / f"{dataset_name.lower()}_parsing_stats.json"
    
    # Tạo thư mục parsed nếu chưa có
    (output_dir / "parsed").mkdir(parents=True, exist_ok=True)
    
    if not log_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file log: {log_file}")
    
    # Khởi tạo parser
    parser = LogParser(
        persistence_dir=str(output_dir / "drain3_state")
    )
    
    # Parse dataset
    start_time = datetime.now()
    results = parser.parse_dataset(
        str(log_file),
        dataset_type=dataset_name,
        max_lines=max_lines,
        output_file=str(output_file)
    )
    end_time = datetime.now()
    
    # Tính thời gian
    duration = (end_time - start_time).total_seconds()
    
    # Lưu statistics
    stats = parser.get_statistics()
    stats['parsing_time_seconds'] = duration
    stats['parsing_time_minutes'] = duration / 60
    stats['dataset'] = dataset_name
    stats['timestamp'] = datetime.now().isoformat()
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # In kết quả
    print(f"\n{'='*70}")
    print(f"KẾT QUẢ PARSING {dataset_name}")
    print(f"{'='*70}")
    print(f"✅ Tổng số entries parsed: {len(results):,}")
    print(f"✅ Success rate: {stats['success_rate']:.2f}%")
    print(f"✅ Unique templates: {stats['unique_templates']:,}")
    print(f"✅ Thời gian parsing: {duration/60:.2f} phút ({duration:.2f} giây)")
    print(f"✅ Output file: {output_file}")
    print(f"✅ Stats file: {stats_file}")
    print(f"{'='*70}\n")
    
    return parser, results, stats


# ============================================================================
# SECTION 2: TEST TOKENIZATION
# ============================================================================

def test_tokenization_word_level():
    """Test word-level tokenization"""
    print("\n" + "="*70)
    print("TEST WORD-LEVEL TOKENIZATION")
    print("="*70)
    
    # Load parsed logs
    parsed_file = Path(__file__).parent.parent / "output" / "parsed" / "hdfs_parsed_sample.json"
    if not parsed_file.exists():
        print("⚠️  File hdfs_parsed_sample.json không tồn tại.")
        print("   Chạy test parsing trước: python3 scripts/test_preprocessing.py --section parsing")
        return
    
    with open(parsed_file, 'r') as f:
        parsed_logs = json.load(f)
    
    # Lấy templates
    templates = [log['template'] for log in parsed_logs[:100] if log.get('template')]
    
    print(f"\n📊 Số lượng templates: {len(templates)}")
    
    word_tokenizer = LogTokenizer(method="word", vocab_size=5000, max_length=128)
    word_tokenizer.fit(templates)
    
    print(f"✅ Vocabulary size: {len(word_tokenizer.word_to_id)}")
    
    # Test tokenize một template
    sample_template = templates[0]
    result = word_tokenizer.tokenize(sample_template)
    
    print(f"\n📝 Sample template: {sample_template[:80]}...")
    print(f"   Tokens: {result['tokens'][:10]}")
    print(f"   Token IDs: {result['token_ids'][:10]}")
    print(f"   Length: {result['length']}")


def test_tokenization_bert():
    """Test BERT tokenization"""
    print("\n" + "="*70)
    print("TEST BERT TOKENIZATION")
    print("="*70)
    
    # Load parsed logs
    parsed_file = Path(__file__).parent.parent / "output" / "parsed" / "hdfs_parsed_sample.json"
    if not parsed_file.exists():
        print("⚠️  File hdfs_parsed_sample.json không tồn tại.")
        print("   Chạy test parsing trước: python3 scripts/test_preprocessing.py --section parsing")
        return
    
    with open(parsed_file, 'r') as f:
        parsed_logs = json.load(f)
    
    templates = [log['template'] for log in parsed_logs[:100] if log.get('template')]
    sample_template = templates[0]
    
    try:
        bert_tokenizer = LogTokenizer(method="bert", max_length=128)
        bert_tokenizer.fit(templates)
        
        result = bert_tokenizer.tokenize(sample_template)
        print(f"\n📝 Sample template: {sample_template[:80]}...")
        print(f"   Token IDs: {result['token_ids'][:15]}")
        print(f"   Attention Mask: {result['attention_mask'][:15]}")
        print(f"   Length: {result['length']}")
    except Exception as e:
        print(f"⚠️  BERT tokenization không khả dụng: {str(e)}")


def test_tokenization():
    """Test tất cả tokenization methods"""
    test_tokenization_word_level()
    test_tokenization_bert()


# ============================================================================
# SECTION 3: TEST EMBEDDING
# ============================================================================

def test_embedding_word2vec():
    """Test Word2Vec embedding"""
    print("\n" + "="*70)
    print("TEST WORD2VEC EMBEDDING")
    print("="*70)
    
    # Load parsed logs
    parsed_file = Path(__file__).parent.parent / "output" / "parsed" / "hdfs_parsed_sample.json"
    if not parsed_file.exists():
        print("⚠️  File hdfs_parsed_sample.json không tồn tại.")
        return
    
    with open(parsed_file, 'r') as f:
        parsed_logs = json.load(f)
    
    templates = [log['template'] for log in parsed_logs[:100] if log.get('template')]
    
    try:
        embedder = LogEmbedder(method="word2vec", embedding_dim=128)
        tokenized_templates = [t.split() for t in templates]
        embedder.fit(templates, tokenized_templates)
        
        print(f"✅ Word2Vec model trained")
        print(f"   Vocabulary size: {len(embedder.model.wv.key_to_index)}")
        
        sample_template = templates[0]
        embedding = embedder.embed_single(sample_template)
        
        print(f"\n📝 Sample template: {sample_template[:80]}...")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding sample: {embedding[:5]}")
    except Exception as e:
        print(f"⚠️  Word2Vec không khả dụng: {str(e)}")


def test_embedding_tfidf():
    """Test TF-IDF embedding"""
    print("\n" + "="*70)
    print("TEST TF-IDF EMBEDDING")
    print("="*70)
    
    # Load parsed logs
    parsed_file = Path(__file__).parent.parent / "output" / "parsed" / "hdfs_parsed_sample.json"
    if not parsed_file.exists():
        print("⚠️  File hdfs_parsed_sample.json không tồn tại.")
        return
    
    with open(parsed_file, 'r') as f:
        parsed_logs = json.load(f)
    
    templates = [log['template'] for log in parsed_logs[:100] if log.get('template')]
    
    try:
        embedder = LogEmbedder(method="tfidf")
        embedder.fit(templates)
        
        sample_template = templates[0]
        embedding = embedder.embed_single(sample_template)
        
        print(f"\n📝 Sample template: {sample_template[:80]}...")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Sparse features: {np.sum(embedding > 0)}/{len(embedding)}")
    except Exception as e:
        print(f"⚠️  TF-IDF không khả dụng: {str(e)}")


def test_embedding_bert():
    """Test BERT embedding"""
    print("\n" + "="*70)
    print("TEST BERT EMBEDDING")
    print("="*70)
    
    # Load parsed logs
    parsed_file = Path(__file__).parent.parent / "output" / "parsed" / "hdfs_parsed_sample.json"
    if not parsed_file.exists():
        print("⚠️  File hdfs_parsed_sample.json không tồn tại.")
        return
    
    with open(parsed_file, 'r') as f:
        parsed_logs = json.load(f)
    
    templates = [log['template'] for log in parsed_logs[:100] if log.get('template')]
    
    try:
        embedder = LogEmbedder(method="bert", device="cpu")
        embedder.fit(templates)
        
        sample_template = templates[0]
        embedding = embedder.embed_single(sample_template)
        
        print(f"\n📝 Sample template: {sample_template[:80]}...")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding sample: {embedding[:5]}")
    except Exception as e:
        print(f"⚠️  BERT không khả dụng: {str(e)}")


def test_embedding():
    """Test tất cả embedding methods"""
    test_embedding_word2vec()
    test_embedding_tfidf()
    test_embedding_bert()


# ============================================================================
# SECTION 4: TEST COMPLETE PIPELINE
# ============================================================================

def test_complete_pipeline():
    """Test complete pipeline: Parsing → Tokenization → Embedding"""
    print("\n" + "="*70)
    print("TEST COMPLETE PIPELINE")
    print("="*70)
    
    # Load parsed logs
    parsed_file = Path(__file__).parent.parent / "output" / "parsed" / "hdfs_parsed_sample.json"
    if not parsed_file.exists():
        print("⚠️  File hdfs_parsed_sample.json không tồn tại.")
        print("   Chạy test parsing trước: python3 scripts/test_preprocessing.py --section parsing")
        return
    
    with open(parsed_file, 'r') as f:
        parsed_logs = json.load(f)[:50]  # Test với 50 logs
    
    print(f"\n📊 Processing {len(parsed_logs)} parsed logs")
    
    # Initialize pipeline
    pipeline = LogPreprocessingPipeline(
        tokenizer_method="word",
        embedder_method="word2vec",
        embedding_dim=128,
        vocab_size=5000
    )
    
    # Fit pipeline
    print("\n🔧 Fitting pipeline...")
    pipeline.fit(parsed_logs)
    
    # Process logs
    print("\n⚙️  Processing logs...")
    processed_logs = pipeline.process_parsed_logs(parsed_logs)
    
    # Statistics
    success_count = sum(1 for r in processed_logs if r.get('embedded') is not None)
    print(f"\n✅ Processed {len(processed_logs)} logs")
    print(f"   Success rate: {success_count/len(processed_logs)*100:.2f}%")
    
    # Show example
    for i, result in enumerate(processed_logs[:3]):
        if result.get('embedded'):
            print(f"\n📝 Example {i+1}:")
            print(f"   Template: {result['template'][:60]}...")
            print(f"   Embedded shape: {len(result['embedded'])}")
            print(f"   Tokenized length: {result['tokenized']['length']}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function với argument parsing"""
    parser = argparse.ArgumentParser(
        description='Test Preprocessing Pipeline - Tất cả test code tập trung ở đây',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test tất cả (với HDFS_2k.log mặc định)
  python3 scripts/test_preprocessing.py

  # Test với file log cụ thể
  python3 scripts/test_preprocessing.py --log_file datasets/HDFS.log

  # Test chỉ parsing
  python3 scripts/test_preprocessing.py --section parsing

  # Parse full dataset (Colab)
  python3 scripts/test_preprocessing.py --parse_full --dataset HDFS
        """
    )
    
    parser.add_argument('--section', type=str,
                       choices=['parsing', 'tokenization', 'embedding', 'pipeline', 'all'],
                       default='all',
                       help='Section to test (default: all)')
    
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to log file (default: datasets/HDFS_2k.log for local, datasets/HDFS.log for Colab)')
    
    parser.add_argument('--dataset', type=str, choices=['HDFS', 'BGL'], default='HDFS',
                       help='Dataset type (default: HDFS)')
    
    parser.add_argument('--max_lines', type=int, default=2000,
                       help='Maximum lines to parse (default: 2000, None for full dataset)')
    
    parser.add_argument('--parse_full', action='store_true',
                       help='Parse full dataset (for Colab)')
    
    args = parser.parse_args()
    
    # Parse full dataset mode
    if args.parse_full:
        print("\n" + "🚀 " + "="*68)
        print(" " * 25 + "FULL DATASET PARSING")
        print("🚀 " + "="*68)
        
        parse_full_dataset(args.dataset, args.max_lines if args.max_lines > 0 else None)
        return
    
    # Normal test mode
    print("\n" + "🔍 " + "="*68)
    print(" " * 20 + "PREPROCESSING PIPELINE TESTING")
    print("🔍 " + "="*68)
    
    if args.section == 'parsing' or args.section == 'all':
        test_parser_single_log()
        test_parser_sample(args.log_file, args.dataset, args.max_lines)
    
    if args.section == 'tokenization' or args.section == 'all':
        test_tokenization()
    
    if args.section == 'embedding' or args.section == 'all':
        test_embedding()
    
    if args.section == 'pipeline' or args.section == 'all':
        test_complete_pipeline()
    
    print("\n" + "="*70)
    print("✅ Tất cả tests hoàn thành!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

