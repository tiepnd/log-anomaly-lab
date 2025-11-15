"""
Log Parser sử dụng Drain3 để parse log raw thành structured logs
Hỗ trợ cả HDFS và BGL datasets
"""

import os
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

try:
    from drain3 import TemplateMiner
    from drain3.file_persistence import FilePersistence
    from drain3.template_miner_config import TemplateMinerConfig
    DRAIN3_AVAILABLE = True
except ImportError:
    DRAIN3_AVAILABLE = False
    print("Warning: drain3 not installed. Please install: pip install drain3")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogParser:
    """
    Log Parser sử dụng Drain3 để trích xuất template và parameters từ log messages
    """
    
    def __init__(self, config_path: str = None, persistence_dir: str = "drain3_state"):
        """
        Khởi tạo LogParser với Drain3
        
        Args:
            config_path: Đường dẫn đến file config của Drain3 (optional)
            persistence_dir: Thư mục lưu state của Drain3 (để resume parsing)
        """
        if not DRAIN3_AVAILABLE:
            raise ImportError("drain3 not installed. Please install: pip install drain3")
        
        # Tạo persistence directory nếu chưa có
        os.makedirs(persistence_dir, exist_ok=True)
        
        # Drain3 configuration - sử dụng TemplateMinerConfig hoặc default
        if config_path is None:
            # Sử dụng default config hoặc tạo config object
            # Drain3 sẽ tự động tìm drain3.ini hoặc dùng default config
            config = None  # Dùng default config
        else:
            # Load config từ file (nếu có file config riêng)
            config = None  # Có thể implement sau nếu cần custom config
        
        # Khởi tạo Drain3 với persistence
        persistence = FilePersistence(os.path.join(persistence_dir, "drain3_state.bin"))
        self.template_miner = TemplateMiner(persistence, config=config)
        
        # Statistics
        self.stats = {
            'total_logs': 0,
            'parsed_logs': 0,
            'failed_logs': 0,
            'templates': defaultdict(int),
            'unique_templates': set()
        }
    
    def extract_message(self, log_line: str, dataset_type: str = "HDFS") -> str:
        """
        Trích xuất log message từ raw log line
        
        Args:
            log_line: Raw log line
            dataset_type: "HDFS" hoặc "BGL"
        
        Returns:
            Log message (phần message sau timestamp, level, component)
        """
        if dataset_type == "HDFS":
            # HDFS format: "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block..."
            # Bỏ qua timestamp (3 phần đầu) và lấy phần còn lại
            parts = log_line.split()
            if len(parts) >= 5:
                # Bỏ qua 3 phần đầu (timestamp) và level, lấy phần còn lại
                message = ' '.join(parts[4:])
                return message
            return log_line
        
        elif dataset_type == "BGL":
            # BGL format: "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
            # Bỏ qua các phần đầu và lấy message sau "RAS KERNEL INFO"
            parts = log_line.split()
            if "RAS" in parts:
                ras_index = parts.index("RAS")
                if ras_index + 3 < len(parts):
                    message = ' '.join(parts[ras_index + 3:])
                    return message
            return log_line
        
        return log_line
    
    def _extract_parameters(self, template: str, original_message: str) -> List[str]:
        """
        Extract parameters từ template và original message
        Template có format: "Connection to <*> failed"
        Original: "Connection to 192.168.1.1 failed"
        Parameters: ["192.168.1.1"]
        """
        if not template or '<*>' not in template:
            return []
        
        # Split template và message
        template_parts = template.split()
        message_parts = original_message.split()
        
        # Tìm parameters (các phần khác nhau giữa template và message)
        parameters = []
        if len(template_parts) == len(message_parts):
            for t_part, m_part in zip(template_parts, message_parts):
                if t_part == '<*>':
                    parameters.append(m_part)
        
        return parameters
    
    def parse_log(self, log_line: str, dataset_type: str = "HDFS") -> Optional[Dict]:
        """
        Parse một log entry
        
        Args:
            log_line: Raw log line
            dataset_type: "HDFS" hoặc "BGL"
        
        Returns:
            Dict chứa template, parameters, template_id hoặc None nếu parse fail
        """
        try:
            # Trích xuất message
            message = self.extract_message(log_line, dataset_type)
            
            # Parse với Drain3
            result = self.template_miner.add_log_message(message)
            
            if result:
                # Drain3 trả về dict với keys: change_type, cluster_id, cluster_size, template_mined, cluster_count
                template_id = result.get('cluster_id', 0)
                template = result.get('template_mined', '')
                # Parameters được extract từ template (các phần <*> trong template)
                # Drain3 không trả về parameters trực tiếp, cần extract từ template
                parameters = self._extract_parameters(result.get('template_mined', ''), message)
                
                # Update statistics
                self.stats['total_logs'] += 1
                self.stats['parsed_logs'] += 1
                self.stats['templates'][template_id] += 1
                self.stats['unique_templates'].add(template_id)
                
                return {
                    'template_id': template_id,
                    'template': template,
                    'parameters': parameters,
                    'original_message': message,
                    'original_log': log_line
                }
            else:
                self.stats['total_logs'] += 1
                self.stats['failed_logs'] += 1
                return None
        
        except Exception as e:
            logger.error(f"Error parsing log: {log_line[:100]}... Error: {str(e)}")
            self.stats['total_logs'] += 1
            self.stats['failed_logs'] += 1
            return None
    
    def parse_dataset(self, log_file: str, dataset_type: str = "HDFS", 
                     max_lines: Optional[int] = None, 
                     output_file: Optional[str] = None) -> List[Dict]:
        """
        Parse toàn bộ dataset
        
        Args:
            log_file: Đường dẫn đến file log
            dataset_type: "HDFS" hoặc "BGL"
            max_lines: Số dòng tối đa để parse (None = parse tất cả, dùng cho testing)
            output_file: Đường dẫn đến file output để lưu kết quả (JSON format)
        
        Returns:
            List các parsed logs
        """
        parsed_logs = []
        total_lines = 0
        
        logger.info(f"Bắt đầu parsing {log_file} (dataset: {dataset_type})")
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if max_lines and line_num > max_lines:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    parsed_result = self.parse_log(line, dataset_type)
                    if parsed_result:
                        parsed_result['line_number'] = line_num
                        parsed_logs.append(parsed_result)
                    
                    total_lines += 1
                    
                    if total_lines % 10000 == 0:
                        logger.info(f"Đã parse {total_lines} dòng...")
            
            # Lưu kết quả nếu có output_file
            if output_file:
                self.save_results(parsed_logs, output_file)
            
            # In statistics
            self.print_statistics()
            
            return parsed_logs
        
        except FileNotFoundError:
            logger.error(f"File không tồn tại: {log_file}")
            return []
        except Exception as e:
            logger.error(f"Error parsing dataset: {str(e)}")
            return []
    
    def save_results(self, parsed_logs: List[Dict], output_file: str):
        """
        Lưu kết quả parsing vào file JSON
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(parsed_logs, f, indent=2, ensure_ascii=False)
            logger.info(f"Đã lưu {len(parsed_logs)} parsed logs vào {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
    
    def print_statistics(self):
        """
        In thống kê parsing
        """
        total = self.stats['total_logs']
        parsed = self.stats['parsed_logs']
        failed = self.stats['failed_logs']
        success_rate = (parsed / total * 100) if total > 0 else 0
        unique_templates = len(self.stats['unique_templates'])
        
        print("\n" + "="*60)
        print("THỐNG KÊ PARSING")
        print("="*60)
        print(f"Tổng số log entries: {total:,}")
        print(f"Parse thành công: {parsed:,} ({success_rate:.2f}%)")
        print(f"Parse thất bại: {failed:,}")
        print(f"Số lượng templates unique: {unique_templates}")
        print(f"Top 10 templates phổ biến nhất:")
        
        # In top 10 templates
        sorted_templates = sorted(self.stats['templates'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]
        for template_id, count in sorted_templates:
            try:
                # Drain3 có thể không có method get_template, dùng template_id thay thế
                template_str = str(template_id)
                # Nếu có method get_template, sử dụng
                if hasattr(self.template_miner, 'get_template'):
                    template_str = self.template_miner.get_template(template_id)
            except:
                template_str = f"Template_{template_id}"
            print(f"  Template {template_id}: {count:,} occurrences")
            print(f"    Pattern: {template_str}")
        
        print("="*60 + "\n")
    
    def get_statistics(self) -> Dict:
        """
        Trả về thống kê dưới dạng dict
        """
        total = self.stats['total_logs']
        parsed = self.stats['parsed_logs']
        success_rate = (parsed / total * 100) if total > 0 else 0
        
        return {
            'total_logs': total,
            'parsed_logs': parsed,
            'failed_logs': self.stats['failed_logs'],
            'success_rate': success_rate,
            'unique_templates': len(self.stats['unique_templates']),
            'top_templates': dict(sorted(self.stats['templates'].items(), 
                                       key=lambda x: x[1], reverse=True)[:10])
        }



