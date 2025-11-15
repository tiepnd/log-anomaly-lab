#!/usr/bin/env python3
"""
Generate Charts cho Chương 3
Tạo charts: throughput_vs_load.png, latency_distribution.png
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
FIGURES_DIR = BASE_DIR / "figures" / "chapter_03"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_throughput_vs_load(output_path: Path):
    """Plot throughput vs load"""
    
    loads = [100, 500, 1000, 2000, 5000]
    ae_throughput = [100, 500, 1000, 1800, 2500]
    lb_throughput = [100, 500, 800, 1500, 2000]
    
    plt.figure(figsize=(10, 6))
    plt.plot(loads, ae_throughput, marker='o', linewidth=2, label='Autoencoder', markersize=8)
    plt.plot(loads, lb_throughput, marker='s', linewidth=2, label='LogBERT', markersize=8)
    plt.plot(loads, loads, '--', color='gray', alpha=0.5, label='Ideal (1:1)', linewidth=1)
    
    plt.xlabel('Load (logs/second)', fontsize=12, fontweight='bold')
    plt.ylabel('Throughput (logs/second processed)', fontsize=12, fontweight='bold')
    plt.title('Throughput vs Load - Pipeline Realtime', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Annotate saturation points
    plt.annotate('Saturation Point\n(Autoencoder)', xy=(2000, 1800), 
                xytext=(2500, 2200), arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))
    plt.annotate('Saturation Point\n(LogBERT)', xy=(2000, 1500), 
                xytext=(2500, 1200), arrowprops=dict(arrowstyle='->', color='orange', lw=1.5),
                fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_latency_distribution(output_path: Path):
    """Plot latency distribution"""
    
    # Simulate latency distribution
    np.random.seed(42)
    
    # Autoencoder latency distribution
    ae_p50 = 250
    ae_p95 = 350
    ae_p99 = 450
    
    # Generate distribution
    ae_mean = ae_p50
    ae_std = (ae_p95 - ae_p50) / 1.645  # Approximate std from P95
    ae_latencies = np.random.normal(ae_mean, ae_std, 10000)
    ae_latencies = np.clip(ae_latencies, 0, ae_p99 * 1.2)
    
    # LogBERT latency distribution
    lb_p50 = 280
    lb_p95 = 400
    lb_p99 = 500
    
    lb_mean = lb_p50
    lb_std = (lb_p95 - lb_p50) / 1.645
    lb_latencies = np.random.normal(lb_mean, lb_std, 10000)
    lb_latencies = np.clip(lb_latencies, 0, lb_p99 * 1.2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Autoencoder
    ax1.hist(ae_latencies, bins=50, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    ax1.axvline(ae_p50, color='green', linestyle='--', linewidth=2, label=f'P50: {ae_p50}ms')
    ax1.axvline(ae_p95, color='orange', linestyle='--', linewidth=2, label=f'P95: {ae_p95}ms')
    ax1.axvline(ae_p99, color='red', linestyle='--', linewidth=2, label=f'P99: {ae_p99}ms')
    ax1.set_xlabel('Latency (ms)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Autoencoder - Latency Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # LogBERT
    ax2.hist(lb_latencies, bins=50, alpha=0.7, color='orange', edgecolor='black', linewidth=0.5)
    ax2.axvline(lb_p50, color='green', linestyle='--', linewidth=2, label=f'P50: {lb_p50}ms')
    ax2.axvline(lb_p95, color='orange', linestyle='--', linewidth=2, label=f'P95: {lb_p95}ms')
    ax2.axvline(lb_p99, color='red', linestyle='--', linewidth=2, label=f'P99: {lb_p99}ms')
    ax2.set_xlabel('Latency (ms)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('LogBERT - Latency Distribution', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def main():
    """Main function"""
    print("=" * 60)
    print("Generating Chapter 3 Charts")
    print("=" * 60)
    
    # Create charts
    print("\n1. Creating throughput vs load chart...")
    throughput_path = FIGURES_DIR / "throughput_vs_load.png"
    plot_throughput_vs_load(throughput_path)
    
    print("\n2. Creating latency distribution chart...")
    latency_path = FIGURES_DIR / "latency_distribution.png"
    plot_latency_distribution(latency_path)
    
    print("\n" + "=" * 60)
    print("✓ All charts generated successfully!")
    print("=" * 60)
    print(f"\nCharts saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()

