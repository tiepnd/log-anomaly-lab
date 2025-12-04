#!/usr/bin/env python3
"""
Script to collect and summarize all results from training, tuning, and evaluation.
Outputs a comprehensive JSON file with all metrics and a summary report.
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def collect_evaluation_results(base_dir: str = "evaluation/output/evaluation") -> Dict:
    """Collect all evaluation results."""
    results = {}
    
    for model_type in ['autoencoder', 'logbert']:
        for dataset in ['HDFS', 'BGL']:
            filepath = f"{base_dir}/{model_type}_{dataset.lower()}_evaluation.json"
            if os.path.exists(filepath):
                results[f"{model_type}_{dataset}"] = load_json(filepath)
    
    return results

def collect_tuning_results(base_dir: str = "training/output/tuning_results") -> Dict:
    """Collect all tuning results."""
    results = {}
    
    for model_type in ['autoencoder', 'logbert']:
        for dataset in ['HDFS', 'BGL']:
            filepath = f"{base_dir}/{model_type}/{dataset.lower()}_tuning_results.json"
            if os.path.exists(filepath):
                data = load_json(filepath)
                # Extract best hyperparameters
                if 'best_params' in data:
                    results[f"{model_type}_{dataset}"] = {
                        'best_params': data['best_params'],
                        'best_score': data.get('best_score', None),
                        'improvement': data.get('improvement', None)
                    }
    
    return results

def collect_thresholds(base_dir: str = "training/output/thresholds") -> Dict:
    """Collect all threshold values."""
    results = {}
    
    for model_type in ['autoencoder', 'logbert']:
        for dataset in ['HDFS', 'BGL']:
            filepath = f"{base_dir}/{model_type}_{dataset.lower()}_threshold.json"
            if os.path.exists(filepath):
                results[f"{model_type}_{dataset}"] = load_json(filepath)
    
    return results

def create_summary_table(eval_results: Dict) -> pd.DataFrame:
    """Create a summary table comparing all models."""
    rows = []
    
    for key, data in eval_results.items():
        model_type, dataset = key.split('_', 1)
        rows.append({
            'Model': model_type.capitalize(),
            'Dataset': dataset,
            'Precision': data.get('precision', 0),
            'Recall': data.get('recall', 0),
            'F1-Score': data.get('f1_score', 0),
            'ROC-AUC': data.get('roc_auc', 0),
            'PR-AUC': data.get('pr_auc', 0),
            'Accuracy': data.get('accuracy', 0),
            'False Positives': data.get('false_positives', 0),
            'False Negatives': data.get('false_negatives', 0),
            'True Positives': data.get('true_positives', 0),
            'True Negatives': data.get('true_negatives', 0)
        })
    
    df = pd.DataFrame(rows)
    return df

def generate_report(results: Dict, output_dir: str = "results_summary"):
    """Generate a comprehensive report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full results
    with open(f"{output_dir}/full_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary table
    if 'evaluation' in results:
        df = create_summary_table(results['evaluation'])
        df.to_csv(f"{output_dir}/summary_table.csv", index=False)
        df.to_markdown(f"{output_dir}/summary_table.md", index=False)
    
    # Generate text report
    report_lines = []
    report_lines.append("# Results Summary Report\n")
    report_lines.append("## Evaluation Results\n\n")
    
    if 'evaluation' in results:
        for key, data in results['evaluation'].items():
            model_type, dataset = key.split('_', 1)
            report_lines.append(f"### {model_type.capitalize()} - {dataset}\n\n")
            report_lines.append(f"- **Precision**: {data.get('precision', 0):.4f}\n")
            report_lines.append(f"- **Recall**: {data.get('recall', 0):.4f}\n")
            report_lines.append(f"- **F1-Score**: {data.get('f1_score', 0):.4f}\n")
            report_lines.append(f"- **ROC-AUC**: {data.get('roc_auc', 0):.4f}\n")
            report_lines.append(f"- **PR-AUC**: {data.get('pr_auc', 0):.4f}\n")
            report_lines.append(f"- **Accuracy**: {data.get('accuracy', 0):.4f}\n\n")
    
    report_lines.append("## Best Hyperparameters\n\n")
    if 'tuning' in results:
        for key, data in results['tuning'].items():
            model_type, dataset = key.split('_', 1)
            report_lines.append(f"### {model_type.capitalize()} - {dataset}\n\n")
            if 'best_params' in data:
                for param, value in data['best_params'].items():
                    report_lines.append(f"- **{param}**: {value}\n")
            if 'best_score' in data and data['best_score']:
                report_lines.append(f"- **Best Score**: {data['best_score']:.4f}\n")
            if 'improvement' in data and data['improvement']:
                report_lines.append(f"- **Improvement**: {data['improvement']:.2f}%\n")
            report_lines.append("\n")
    
    report_lines.append("## Thresholds\n\n")
    if 'thresholds' in results:
        for key, data in results['thresholds'].items():
            model_type, dataset = key.split('_', 1)
            threshold = data.get('threshold', 'N/A')
            report_lines.append(f"- **{model_type.capitalize()} - {dataset}**: {threshold}\n")
    
    with open(f"{output_dir}/report.md", 'w') as f:
        f.writelines(report_lines)
    
    print(f"✓ Results collected and saved to {output_dir}/")
    print(f"  - full_results.json: Complete results")
    print(f"  - summary_table.csv: Summary table (CSV)")
    print(f"  - summary_table.md: Summary table (Markdown)")
    print(f"  - report.md: Text report")

def main():
    """Main function."""
    print("Collecting results...")
    
    results = {
        'evaluation': collect_evaluation_results(),
        'tuning': collect_tuning_results(),
        'thresholds': collect_thresholds()
    }
    
    generate_report(results)
    
    # Print summary
    print("\n=== SUMMARY ===")
    print(f"Evaluation results: {len(results['evaluation'])} models")
    print(f"Tuning results: {len(results['tuning'])} models")
    print(f"Thresholds: {len(results['thresholds'])} models")

if __name__ == "__main__":
    main()

