#!/usr/bin/env python3
"""
TensorRT Benchmark Visualization

Creates visualization charts for benchmark results.
Supports both standard and dynamic batch benchmark results.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import numpy as np
import matplotlib.cm as cm
from typing import Dict, Tuple
import sys
import os


def is_dynamic_batch(data: Dict) -> bool:
    """Check if benchmark results are from dynamic batch mode."""
    return data.get('test_info', {}).get('test_type') == 'dynamic_batch'


def extract_data_for_dynamic_batch(data: Dict) -> Tuple:
    """
    Extract and organize data for dynamic batch visualization.

    Returns:
        plan_batches: sorted list of plan batch sizes
        infer_batches: dict {plan_batch: list of infer_batches}
        streams: sorted list of stream counts
        throughput_dict: dict {(plan_batch, infer_batch, stream): throughput}
    """
    results = data['all_results']

    plan_batches = sorted(list(set(r['plan_batch'] for r in results)))
    streams = sorted(list(set(r['num_streams'] for r in results)))

    # Build infer_batches mapping
    infer_batches = {}
    for plan_batch in plan_batches:
        infer_list = sorted(list(set(r['infer_batch'] for r in results if r['plan_batch'] == plan_batch)))
        infer_batches[plan_batch] = infer_list

    # Build throughput dictionary
    throughput_dict = {}
    for r in results:
        key = (r['plan_batch'], r['infer_batch'], r['num_streams'])
        throughput_dict[key] = r['throughput_nneval']

    return plan_batches, infer_batches, streams, throughput_dict


def create_single_plan_curves_gif(data: Dict, output_path: str, fps: int = 1, dpi: int = 150):
    """
    Create animated GIF showing performance curves for each plan batch.
    Each frame shows one plan_batch with lines for different stream counts.
    """
    plan_batches, infer_batches, streams, throughput_dict = extract_data_for_dynamic_batch(data)
    gpu_name = data.get('test_info', {}).get('gpu', 'Unknown GPU')

    # Use viridis colormap for stream counts
    colors = cm.viridis(np.linspace(0.2, 0.9, len(streams)))

    fig, ax = plt.subplots(figsize=(12, 8))

    def update_frame(frame_idx):
        ax.clear()
        plan_batch = plan_batches[frame_idx]
        infer_list = infer_batches[plan_batch]

        # Plot lines for each stream count
        for i, stream in enumerate(streams):
            throughputs = [throughput_dict.get((plan_batch, infer_b, stream), 0)
                          for infer_b in infer_list]
            line = ax.plot(infer_list, throughputs, 'o-',
                          label=f'{stream} Stream(s)',
                          color=colors[i], linewidth=2, markersize=6)

            # Mark the maximum point
            if throughputs:
                max_idx = np.argmax(throughputs)
                max_val = throughputs[max_idx]
                ax.plot(infer_list[max_idx], max_val, '*',
                       color=colors[i], markersize=15, markeredgecolor='red', markeredgewidth=2)

        ax.set_xlabel('Inference Batch Size', fontsize=14)
        ax.set_ylabel('Throughput (nnEval/s)', fontsize=14)
        ax.set_title(f'Plan Batch = {plan_batch} - {gpu_name}', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0, right=max(infer_list) * 1.05)
        ax.set_ylim(bottom=0)

    anim = animation.FuncAnimation(fig, update_frame,
                                   frames=len(plan_batches),
                                   interval=1000//fps, repeat=True)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def create_cross_plan_comparison_gif(data: Dict, output_path: str, fps: int = 1, dpi: int = 150):
    """
    Create animated GIF comparing different plans at same inference batch.
    Each frame shows one infer_batch with grouped bars for different streams.
    """
    plan_batches, infer_batches, streams, throughput_dict = extract_data_for_dynamic_batch(data)
    gpu_name = data.get('test_info', {}).get('gpu', 'Unknown GPU')

    # Get all unique infer_batch values
    all_infer_batches = sorted(list(set(
        infer_b for infer_list in infer_batches.values() for infer_b in infer_list
    )))

    # Use viridis colormap for stream counts
    colors = cm.viridis(np.linspace(0.2, 0.9, len(streams)))

    fig, ax = plt.subplots(figsize=(14, 8))

    def update_frame(frame_idx):
        ax.clear()
        infer_batch = all_infer_batches[frame_idx]

        # Prepare data for grouped bar chart
        x = np.arange(len(plan_batches))
        width = 0.8 / len(streams)

        for i, stream in enumerate(streams):
            throughputs = []
            for plan_batch in plan_batches:
                # Only include if this infer_batch is valid for this plan_batch
                if infer_batch <= plan_batch:
                    tp = throughput_dict.get((plan_batch, infer_batch, stream), 0)
                    throughputs.append(tp)
                else:
                    throughputs.append(0)

            offset = (i - len(streams)/2 + 0.5) * width
            bars = ax.bar(x + offset, throughputs, width,
                         label=f'{stream} Stream(s)', color=colors[i])

            # Add value labels on bars
            for j, (bar, val) in enumerate(zip(bars, throughputs)):
                if val > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(val)}',
                           ha='center', va='bottom', fontsize=8, rotation=90)

        ax.set_xlabel('Plan Batch Size', fontsize=14)
        ax.set_ylabel('Throughput (nnEval/s)', fontsize=14)
        ax.set_title(f'Infer Batch = {infer_batch} - {gpu_name}', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(plan_batches)
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(bottom=0)

    anim = animation.FuncAnimation(fig, update_frame,
                                   frames=len(all_infer_batches),
                                   interval=1000//fps, repeat=True)
    anim.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def create_max_performance_lines(data: Dict, output_path: str, dpi: int = 150):
    """
    Create static line plot showing max performance (infer_batch = plan_batch).
    4 lines for 4 stream counts.
    """
    gpu_name = data.get('test_info', {}).get('gpu', 'Unknown GPU')

    if is_dynamic_batch(data):
        plan_batches, infer_batches, streams, throughput_dict = extract_data_for_dynamic_batch(data)

        # Extract throughput where infer_batch == plan_batch
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = cm.viridis(np.linspace(0.2, 0.9, len(streams)))

        max_throughput = 0
        max_point = None

        for i, stream in enumerate(streams):
            throughputs = []
            for plan_batch in plan_batches:
                tp = throughput_dict.get((plan_batch, plan_batch, stream), 0)
                throughputs.append(tp)
                if tp > max_throughput:
                    max_throughput = tp
                    max_point = (plan_batch, tp, stream)

            ax.plot(plan_batches, throughputs, 'o-',
                   label=f'{stream} Stream(s)',
                   color=colors[i], linewidth=2, markersize=6)

        # Annotate maximum point
        if max_point:
            ax.plot(max_point[0], max_point[1], '*',
                   markersize=20, color='red', markeredgecolor='darkred', markeredgewidth=2)
            ax.annotate(f'Max: {int(max_point[1])} nnEval/s\n(batch={max_point[0]}, streams={max_point[2]})',
                       xy=(max_point[0], max_point[1]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        ax.set_xlabel('Plan Batch Size', fontsize=14)
        ax.set_ylabel('Throughput (nnEval/s)', fontsize=14)
        ax.set_title(f'Maximum Performance (Infer Batch = Plan Batch) - {gpu_name}',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    else:
        # Standard mode: similar to original create_throughput_plot
        results = data['all_results']
        batch_sizes = sorted(list(set(r['batch_size'] for r in results)))
        streams = sorted(list(set(r['num_streams'] for r in results)))

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = cm.viridis(np.linspace(0.2, 0.9, len(streams)))

        for i, stream in enumerate(streams):
            throughputs = []
            for batch in batch_sizes:
                matching = [r for r in results
                           if r['batch_size'] == batch and r['num_streams'] == stream]
                if matching:
                    throughputs.append(matching[0]['throughput_nneval'])
                else:
                    throughputs.append(0)

            ax.plot(batch_sizes, throughputs, 'o-',
                   label=f'{stream} Stream(s)',
                   color=colors[i], linewidth=2, markersize=6)

        ax.set_xlabel('Batch Size', fontsize=14)
        ax.set_ylabel('Throughput (nnEval/s)', fontsize=14)
        ax.set_title(f'TensorRT Throughput vs Batch Size - {gpu_name}',
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def generate_best_config_table(data: Dict, output_path: str):
    """
    Generate text file with best configuration recommendations.
    """
    results = data['all_results']
    gpu_name = data.get('test_info', {}).get('gpu', 'Unknown GPU')

    with open(output_path, 'w') as f:
        f.write("Best Configuration Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"GPU: {gpu_name}\n\n")

        if is_dynamic_batch(data):
            plan_batches, infer_batches, streams, throughput_dict = extract_data_for_dynamic_batch(data)

            # Get all unique infer_batch values
            all_infer_batches = sorted(list(set(
                infer_b for infer_list in infer_batches.values() for infer_b in infer_list
            )))

            f.write("For each target inference batch size, recommended configuration:\n\n")
            f.write(f"{'Target Infer':<15} | {'Best Plan':<12} | {'Best Streams':<13} | {'Throughput':<20} | Notes\n")
            f.write(f"{'Batch':<15} | {'Batch':<12} | {'':<13} | {'(nnEval/s)':<20} |\n")
            f.write("-" * 90 + "\n")

            # Find best config for each infer_batch
            best_overall = None
            best_overall_throughput = 0

            for infer_batch in all_infer_batches:
                best_tp = 0
                best_plan = None
                best_stream = None

                for plan_batch in plan_batches:
                    if infer_batch <= plan_batch:
                        for stream in streams:
                            tp = throughput_dict.get((plan_batch, infer_batch, stream), 0)
                            if tp > best_tp:
                                best_tp = tp
                                best_plan = plan_batch
                                best_stream = stream

                if best_plan:
                    note = ""
                    if best_plan != infer_batch:
                        note = f"Using plan_batch={best_plan}"

                    f.write(f"{infer_batch:<15} | {best_plan:<12} | {best_stream:<13} | {best_tp:<20.0f} | {note}\n")

                    if best_tp > best_overall_throughput:
                        best_overall_throughput = best_tp
                        best_overall = (best_plan, infer_batch, best_stream, best_tp)

            f.write("\n")
            if best_overall:
                f.write(f"Overall Best: plan_batch={best_overall[0]}, infer_batch={best_overall[1]}, ")
                f.write(f"streams={best_overall[2]} -> {best_overall[3]:.0f} nnEval/s\n")

            # Add regression check section
            f.write("\n" + "=" * 80 + "\n")
            f.write("Performance at near-max batch sizes (checking regression):\n\n")

            for plan_batch in plan_batches:
                infer_list = sorted(infer_batches[plan_batch], reverse=True)[:5]  # Top 5
                if len(infer_list) >= 2:
                    f.write(f"\nPlan Batch {plan_batch}:\n")
                    f.write(f"{'Infer Batch':<13} | {'Streams':<10} | {'Throughput':<15} | % of Max\n")
                    f.write("-" * 60 + "\n")

                    # Find max for this plan
                    max_tp = 0
                    for infer_b in infer_list:
                        for stream in streams:
                            tp = throughput_dict.get((plan_batch, infer_b, stream), 0)
                            if tp > max_tp:
                                max_tp = tp

                    for infer_b in infer_list:
                        best_stream_tp = 0
                        best_stream_for_infer = None
                        for stream in streams:
                            tp = throughput_dict.get((plan_batch, infer_b, stream), 0)
                            if tp > best_stream_tp:
                                best_stream_tp = tp
                                best_stream_for_infer = stream

                        if best_stream_for_infer and max_tp > 0:
                            pct = (best_stream_tp / max_tp) * 100
                            f.write(f"{infer_b:<13} | {best_stream_for_infer:<10} | {best_stream_tp:<15.0f} | {pct:.1f}%\n")

        else:
            # Standard mode
            f.write("Top configurations by throughput:\n\n")
            sorted_results = sorted(results, key=lambda x: x['throughput_nneval'], reverse=True)

            f.write(f"{'Rank':<6} | {'Batch':<8} | {'Streams':<10} | {'Throughput (nnEval/s)':<20}\n")
            f.write("-" * 60 + "\n")

            for i, r in enumerate(sorted_results[:10], 1):
                f.write(f"{i:<6} | {r['batch_size']:<8} | {r['num_streams']:<10} | {r['throughput_nneval']:<20.0f}\n")

            if sorted_results:
                best = sorted_results[0]
                f.write(f"\nBest: batch={best['batch_size']}, streams={best['num_streams']} ")
                f.write(f"-> {best['throughput_nneval']:.0f} nnEval/s\n")


def visualize_results(data: Dict, output_dir: str, fps: int = 1, dpi: int = 150):
    """Main visualization function."""
    os.makedirs(output_dir, exist_ok=True)

    if is_dynamic_batch(data):
        print("Detected dynamic batch benchmark results")
        print("Generating visualizations...")

        print("  [1/4] Single plan curves animation...")
        create_single_plan_curves_gif(data, f"{output_dir}/1_single_plan_curves.gif", fps, dpi)

        print("  [2/4] Cross plan comparison animation...")
        create_cross_plan_comparison_gif(data, f"{output_dir}/2_cross_plan_comparison.gif", fps, dpi)

        print("  [3/4] Max performance lines...")
        create_max_performance_lines(data, f"{output_dir}/3_max_performance_lines.png", dpi)

        print("  [4/4] Best config summary...")
        generate_best_config_table(data, f"{output_dir}/4_best_config_summary.txt")
    else:
        print("Detected standard benchmark results")
        print("Generating visualization...")
        create_max_performance_lines(data, f"{output_dir}/throughput_chart.png", dpi)
        generate_best_config_table(data, f"{output_dir}/best_config_summary.txt")

    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize TensorRT benchmark results')
    parser.add_argument('-f', '--file', required=True,
                       help='Path to benchmark_results.json')
    parser.add_argument('-o', '--output-dir', default='./visualizations',
                       help='Output directory (default: ./visualizations)')
    parser.add_argument('--fps', type=int, default=1,
                       help='Frames per second for GIF animations (default: 1)')
    parser.add_argument('--dpi', type=int, default=150,
                       help='DPI for output images (default: 150)')

    args = parser.parse_args()

    # Load data
    try:
        with open(args.file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {args.file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        sys.exit(1)

    visualize_results(data, args.output_dir, args.fps, args.dpi)
