#!/usr/bin/env python3
"""
TensorRT Benchmark Visualization

Creates visualization charts for benchmark results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from typing import Dict, Optional


def create_throughput_plot(data: Dict, gpu_name: str, output_file: str = None) -> None:
    """
    Create line plot showing throughput vs batch size grouped by stream count.

    Args:
        data: Benchmark data dictionary with 'all_results' key
        gpu_name: GPU name to display in title
        output_file: Optional output file path. If None, displays interactively.
    """
    results = data['all_results']

    # Extract data
    batch_sizes = []
    num_streams = []
    throughput_nneval = []

    for result in results:
        batch_sizes.append(result['batch_size'])
        num_streams.append(result['num_streams'])
        throughput_nneval.append(result['throughput_nneval'])

    batch_sizes = np.array(batch_sizes)
    num_streams = np.array(num_streams)
    throughput_nneval = np.array(throughput_nneval)

    # Get unique stream counts and batch sizes
    unique_streams = sorted(list(set(num_streams)))
    unique_batches = sorted(list(set(batch_sizes)))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use viridis colormap for progressive color scheme
    colors = cm.viridis(np.linspace(0.2, 0.9, len(unique_streams)))

    # Plot line for each stream count
    for i, stream in enumerate(unique_streams):
        stream_throughput = []
        stream_batches = []

        for batch in unique_batches:
            mask = (batch_sizes == batch) & (num_streams == stream)
            if np.any(mask):
                stream_throughput.append(throughput_nneval[mask][0])
                stream_batches.append(batch)

        if stream_throughput:
            ax.plot(stream_batches, stream_throughput, 'o-',
                    label=f'{stream} Stream(s)', color=colors[i], linewidth=2, markersize=6)

    ax.set_xlabel('Batch Size', fontsize=14)
    ax.set_ylabel('Throughput (nnEval/s)', fontsize=14)
    ax.set_title(f'TensorRT Throughput vs Batch Size - {gpu_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Set x-axis to show only integer batch sizes
    ax.set_xlim(left=0)
    ax.set_xticks(unique_batches)

    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def print_top_configurations(data: Dict, top_n: int = 10) -> None:
    """
    Print top N configurations by throughput.

    Args:
        data: Benchmark data dictionary
        top_n: Number of top configurations to print
    """
    results = data['all_results']
    # Sort by throughput (nnEval/s)
    sorted_results = sorted(results, key=lambda x: x['throughput_nneval'], reverse=True)

    print(f"Top {top_n} Best Configurations:")
    print("-" * 90)
    print(f"{'Rank':<6} {'Batch Size':<12} {'Streams':<10} {'Throughput':<15} {'Latency(ms)':<12} {'QPS':<15}")
    print("-" * 90)

    for i, result in enumerate(sorted_results[:top_n]):
        print(f"{i+1:<6} {result['batch_size']:<12} {result['num_streams']:<10} "
              f"{result['throughput_nneval']:<15.1f} {result['latency_ms']:<12.3f} "
              f"{result['throughput_qps']:<15.1f}")

    print()
    print(f"Best Overall Configuration:")
    best = sorted_results[0]
    print(f"  Batch Size: {best['batch_size']}")
    print(f"  Number of Streams: {best['num_streams']}")
    print(f"  Throughput: {best['throughput_nneval']:.1f} nnEval/s")
    print(f"  Latency: {best['latency_ms']:.3f} ms")
    print(f"  QPS: {best['throughput_qps']:.1f}")


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Visualize TensorRT benchmark results'
    )
    parser.add_argument('--file', '-f', required=True,
                       help='Path to benchmark_results.json file')
    parser.add_argument('--output', '-o',
                       help='Output file path for chart (default: display interactively)')
    parser.add_argument('--top', '-t', type=int, default=10,
                       help='Number of top configurations to display (default: 10)')

    args = parser.parse_args()

    # Load benchmark data
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {args.file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {e}")
        sys.exit(1)

    # Extract GPU name from test_info
    gpu_name = data.get('test_info', {}).get('gpu', 'Unknown GPU')

    # Print top configurations
    print_top_configurations(data, args.top)
    print()

    # Create visualization
    create_throughput_plot(data, gpu_name, args.output)
