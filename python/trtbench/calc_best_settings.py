#!/usr/bin/env python3
"""
KataGo Optimal Settings Calculator

Calculates optimal KataGo configuration based on benchmark results.
Uses ELO calculation model to evaluate performance of different configurations.
"""

import json
import math
from typing import Dict, List, Tuple


class KataGoSettingsCalculator:
    """KataGo Settings Calculator"""

    def __init__(self):
        self.elo_gain_per_doubling = 250.0  # ELO gain per speed doubling
        self.base_thread_cost = 7.0         # Base thread cost
        self.reference_batch_size = 4       # Reference batch size for relative calculation
        self.reference_num_streams = 1      # Reference num streams for relative calculation

    def calculate_num_search_threads(self, batch_size: int, num_streams: int, num_gpus: int = 1) -> int:
        """Calculate search threads: numSearchThreads = batch_size * (num_streams + 1) * num_gpus"""
        return batch_size * (num_streams + 1) * num_gpus

    def calculate_elo_effect(self,
                           throughput_qps: float,
                           batch_size: int,
                           num_search_threads: int,
                           time_per_move: float,
                           reference_computation_per_move: float = None) -> Dict:
        """Calculate ELO effect based on KataGo's ELO model"""
        visits_per_second = throughput_qps * batch_size  # Total visits per second
        visits_per_move = visits_per_second * time_per_move

        # Calculate total computation per move (qps * batch_size * time_per_move)
        total_computation_per_move = throughput_qps * batch_size * time_per_move

        # Calculate relative computation compared to reference configuration
        if reference_computation_per_move and reference_computation_per_move > 0:
            computation_ratio = total_computation_per_move / reference_computation_per_move
        else:
            computation_ratio = 1.0

        # ELO gain from computation doubling (relative to reference)
        if computation_ratio <= 0:
            gain = 0
        else:
            gain = self.elo_gain_per_doubling * math.log(computation_ratio, 2)

        # ELO cost from threads and visits
        if visits_per_move <= 0:
            cost = 0
            adjustment_factor = 1.0
        else:
            adjustment_factor = (1600.0 / (800.0 + visits_per_move)) ** 0.85
            cost = num_search_threads * self.base_thread_cost * adjustment_factor

        return {
            'gain': gain,
            'cost': cost,
            'adjustment_factor': adjustment_factor,
            'net_effect': gain - cost,
            'computation_ratio': computation_ratio,
            'computation_per_second': throughput_qps * batch_size,
            'computation_per_move': total_computation_per_move,
            'visits_per_second': visits_per_second,
            'visits_per_move': visits_per_move
        }

    def calculate_elo_for_config(self,
                                config: Dict,
                                time_per_move: float,
                                num_gpus: int = 1,
                                reference_computation_per_move: float = None) -> Dict:
        """Calculate ELO metrics for a single configuration"""
        batch_size = config['batch_size']
        num_streams = config['num_streams']
        throughput_qps = config['throughput_qps']

        num_search_threads = self.calculate_num_search_threads(batch_size, num_streams, num_gpus)
        elo_data = self.calculate_elo_effect(throughput_qps, batch_size, num_search_threads, time_per_move, reference_computation_per_move)

        result = config.copy()
        result['num_search_threads'] = num_search_threads
        result['elo_gain'] = elo_data['gain']
        result['elo_cost'] = elo_data['cost']
        result['adjustment_factor'] = elo_data['adjustment_factor']
        result['elo_effect'] = elo_data['net_effect']
        result['computation_ratio'] = elo_data['computation_ratio']
        result['computation_per_second'] = elo_data['computation_per_second']
        result['computation_per_move'] = elo_data['computation_per_move']
        result['visits_per_second'] = elo_data['visits_per_second']
        result['visits_per_move'] = elo_data['visits_per_move']

        return result

    def find_best_config(self,
                        results: List[Dict],
                        time_per_move: float,
                        num_gpus: int = 1) -> Tuple[Dict, List[Dict]]:
        """Find optimal configuration"""
        # Find reference configuration (batch_size=4, num_streams=1)
        reference_config = None
        reference_computation_per_move = None

        for config in results:
            if (config['batch_size'] == self.reference_batch_size and
                config['num_streams'] == self.reference_num_streams):
                reference_config = config
                reference_computation_per_move = config['throughput_qps'] * config['batch_size'] * time_per_move
                break

        if not reference_config:
            print(f"Warning: Reference configuration (batch={self.reference_batch_size}, streams={self.reference_num_streams}) not found")
            print("Using first configuration as reference")
            if results:
                reference_config = results[0]
                reference_computation_per_move = reference_config['throughput_qps'] * reference_config['batch_size'] * time_per_move
            else:
                reference_computation_per_move = 1.0

        elo_results = []
        for config in results:
            elo_config = self.calculate_elo_for_config(config, time_per_move, num_gpus, reference_computation_per_move)
            elo_config['num_gpus'] = num_gpus
            elo_results.append(elo_config)

        best_config = max(elo_results, key=lambda x: x['elo_effect'])
        return best_config, elo_results

    def format_results(self,
                      best_config: Dict,
                      elo_results: List[Dict],
                      time_per_move: float,
                      num_gpus: int) -> str:
        """Format calculation results as string"""
        lines = []
        lines.append("=" * 60)
        lines.append("KataGo Optimal Settings")
        lines.append("=" * 60)
        lines.append(f"Time per move: {time_per_move:.1f}s, GPUs: {num_gpus}")
        lines.append("")
        lines.append("Best Configuration:")
        lines.append(f"  Batch size: {best_config['batch_size']}")
        lines.append(f"  Streams: {best_config['num_streams']}")
        lines.append(f"  Search threads: {best_config['num_search_threads']}")
        lines.append(f"  Latency: {best_config['latency_ms']:.3f}ms")
        lines.append(f"  nnEval/s: {best_config['visits_per_second']:.0f}")
        lines.append(f"  nnEval/move: {best_config['visits_per_move']:.0f}")
        lines.append(f"  ELO gain: {best_config['elo_gain']:+.1f}")
        lines.append(f"  Cost per thread: {best_config['elo_cost']/best_config['num_search_threads']:.2f}")
        lines.append(f"  Net ELO effect: {best_config['elo_effect']:+.1f}")
        lines.append("")
        lines.append("Top 20 Configurations (by ELO effect):")
        lines.append(f"{'Rank':<5} {'Batch':<6} {'Streams':<8} {'Threads':<8} {'Lat(ms)':<8} {'QPS':<6} {'nnEval/s':<9} {'nnEval':<9} {'Gain':<7} {'Cost':<8} {'Cost/thr':<9} {'ELO':<7}")
        lines.append("-" * 106)

        sorted_results = sorted(elo_results, key=lambda x: x['elo_effect'], reverse=True)
        for i, config in enumerate(sorted_results[:20], 1):
            cost_per_thread = config['elo_cost'] / config['num_search_threads']
            lines.append(f"{i:<5} {config['batch_size']:<6} {config['num_streams']:<8} "
                        f"{config['num_search_threads']:<8} {config['latency_ms']:<8.3f} "
                        f"{config['throughput_qps']:<6.0f} {config['visits_per_second']:<9.0f} {config['visits_per_move']:<9.0f} "
                        f"{config['elo_gain']:<+7.1f} {-config['elo_cost']:<+8.1f} "
                        f"{cost_per_thread:<9.2f} {config['elo_effect']:<+7.1f}")
        lines.append("=" * 60)

        return "\n".join(lines)


if __name__ == '__main__':
    import sys
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='KataGo Optimal Settings Calculator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calc_best_settings.py --file benchmark_results.json --time 3.0 --gpus 1
  python calc_best_settings.py --file benchmark_results.json --time 1.5
  python calc_best_settings.py --file benchmark_results.json --time 5.0 --gpus 2
        """
    )

    parser.add_argument('--file', '-f', required=True,
                       help='Path to benchmark_results JSON file')
    parser.add_argument('--time', '-t', type=float, default=3.0,
                       help='Time per move in seconds (default: 3.0)')
    parser.add_argument('--gpus', '-g', type=int, default=1,
                       help='Number of GPUs (default: 1)')

    args = parser.parse_args()

    # Load benchmark results
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            benchmark_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {args.file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file - {e}")
        sys.exit(1)

    results = benchmark_data['all_results']

    # Calculate best settings
    calculator = KataGoSettingsCalculator()
    best_config, elo_results = calculator.find_best_config(
        results, args.time, args.gpus
    )

    # Print results
    output = calculator.format_results(best_config, elo_results, args.time, args.gpus)
    print(output)
