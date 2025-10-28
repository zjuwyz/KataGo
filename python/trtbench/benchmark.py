#!/usr/bin/env python3
"""
TensorRT Benchmark Suite for KataGo

Integrated benchmark pipeline that:
1. Generates TensorRT plan files for different batch sizes
2. Benchmarks each plan with different CUDA stream counts
3. Generates results, visualizations, and optimal settings after each batch size
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from plan_generator import PlanGenerator
from get_plan_filename import PlanFilenameGenerator
from trtexec_runner import TRTExecRunner
from calc_best_settings import KataGoSettingsCalculator
from visualize_benchmark import create_throughput_plot


class TRTBenchmarkSuite:
    """Complete TensorRT benchmark suite"""

    def __init__(
        self,
        katago_executable: str,
        model_file: str,
        trtexec_path: str,
        tensorrt_lib_path: str,
        cache_dir: str,
        gpu_name: str
    ):
        """
        Initialize benchmark suite.

        Args:
            katago_executable: Path to KataGo binary
            model_file: Path to model file
            trtexec_path: Path to trtexec binary
            tensorrt_lib_path: Path to TensorRT libraries
            cache_dir: TensorRT cache directory
            gpu_name: GPU device name
        """
        self.plan_gen = PlanGenerator(katago_executable, model_file, tensorrt_lib_path, cache_dir)
        self.plan_filename_gen = PlanFilenameGenerator(
            os.path.join(tensorrt_lib_path, "libnvinfer.so.10"),
            cache_dir
        )
        self.trt_runner = TRTExecRunner(trtexec_path, tensorrt_lib_path)
        self.settings_calc = KataGoSettingsCalculator()
        self.gpu_name = gpu_name
        self.cache_dir = cache_dir

    def run_benchmark(
        self,
        batch_sizes: List[int],
        stream_counts: List[int],
        duration: float = 5.0,
        warmup: int = 500,
        time_per_move: float = 3.0,
        num_gpus: int = 1
    ) -> str:
        """
        Run complete benchmark suite.

        Args:
            batch_sizes: List of batch sizes to test
            stream_counts: List of stream counts to test
            duration: Benchmark duration per test (seconds)
            warmup: Warmup time (milliseconds)
            time_per_move: Time per move for ELO calculation (seconds)
            num_gpus: Number of GPUs for ELO calculation

        Returns:
            Path to results directory
        """
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path.cwd() / f"benchmark_{timestamp}"
        results_dir.mkdir(exist_ok=True)

        # Create logs subdirectory
        logs_dir = results_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        print("=" * 80)
        print("TensorRT Benchmark Suite for KataGo")
        print("=" * 80)
        print(f"GPU: {self.gpu_name}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Stream counts: {stream_counts}")
        print(f"Duration: {duration}s per test (warmup: {warmup}ms)")
        print(f"Results directory: {results_dir}")
        print("=" * 80)
        print()

        all_results = []
        total_tests = len(batch_sizes) * len(stream_counts)
        current_test = 0

        for batch_size in batch_sizes:
            print(f"\n{'='*80}")
            print(f"Batch Size: {batch_size}")
            print(f"{'='*80}")

            # Step 1: Check if plan file exists, generate if needed
            print(f"\n[1/3] Checking TensorRT plan for batch size {batch_size}...")

            # Try to find existing plan file by pattern matching
            import glob
            cache_pattern = str(Path(self.cache_dir) / f"trt-*_batch{batch_size}_fp16")
            existing_plans = glob.glob(cache_pattern)

            if existing_plans:
                # Use the first matching plan file
                plan_path = existing_plans[0]
                file_size = os.path.getsize(plan_path) / (1024 * 1024)
                print(f"  ✓ Plan file found: {Path(plan_path).name}")
                print(f"    Size: {file_size:.1f} MB")
            else:
                print(f"  Plan file not found, generating...")
                success = self.plan_gen.generate_plan(batch_size)
                if not success:
                    print(f"  ✗ Failed to generate plan for batch size {batch_size}")
                    continue

                # Find the newly created plan file
                existing_plans = glob.glob(cache_pattern)
                if not existing_plans:
                    print(f"  ✗ Plan file not found after generation")
                    continue

                plan_path = existing_plans[0]
                file_size = os.path.getsize(plan_path) / (1024 * 1024)
                print(f"  ✓ Plan generated: {Path(plan_path).name}")
                print(f"    Size: {file_size:.1f} MB")

            # Step 2: Benchmark with different stream counts
            print(f"\n[2/3] Benchmarking batch size {batch_size} with different stream counts...")
            batch_results = []

            for num_streams in stream_counts:
                current_test += 1
                print(f"\n  Test {current_test}/{total_tests}: batch={batch_size}, streams={num_streams}")

                # Create log file path in logs subdirectory
                log_file = logs_dir / f"trtexec_batch{batch_size}_stream{num_streams}.log"

                result = self.trt_runner.benchmark(
                    engine_path=plan_path,
                    batch_size=batch_size,
                    num_streams=num_streams,
                    duration=duration,
                    warmup=warmup,
                    log_file=str(log_file)
                )

                if result:
                    # Calculate throughput metric (nnEval/s)
                    throughput_nneval = batch_size * result['throughput_qps']

                    # Store relative path to log file
                    log_relative = f"logs/{log_file.name}"

                    result_entry = {
                        'batch_size': batch_size,
                        'num_streams': num_streams,
                        'latency_ms': result['latency_ms'],
                        'throughput_qps': result['throughput_qps'],
                        'throughput_nneval': throughput_nneval,
                        'trtexec_command': result['trtexec_command'],
                        'trtexec_log': log_relative
                    }
                    batch_results.append(result_entry)
                    all_results.append(result_entry)

                    print(f"    Latency: {result['latency_ms']:.3f}ms, "
                          f"Throughput: {result['throughput_qps']:.1f}qps, "
                          f"nnEval/s: {throughput_nneval:.0f}")
                    print(f"    Log: {log_relative}")
                else:
                    print(f"    ✗ Benchmark failed")

            # Step 3: Generate results for current batch size
            if batch_results:
                print(f"\n[3/3] Generating results for batch size {batch_size}...")
                self._generate_results(
                    all_results,
                    results_dir,
                    batch_size,
                    time_per_move,
                    num_gpus
                )

        print("\n" + "=" * 80)
        print(f"Benchmark complete! Results saved to: {results_dir}")
        print("=" * 80)

        return str(results_dir)

    def _generate_results(
        self,
        all_results: List[Dict],
        results_dir: Path,
        current_batch: int,
        time_per_move: float,
        num_gpus: int
    ) -> None:
        """
        Generate benchmark results, visualization, and optimal settings.

        Args:
            all_results: All benchmark results so far
            results_dir: Directory to save results
            current_batch: Current batch size being processed
            time_per_move: Time per move for ELO calculation
            num_gpus: Number of GPUs
        """
        # Prepare metadata
        test_info = {
            "gpu": self.gpu_name,
            "total_tests": len(all_results),
            "batch_sizes_tested": sorted(list(set(r['batch_size'] for r in all_results))),
            "stream_counts_tested": sorted(list(set(r['num_streams'] for r in all_results))),
        }

        # Create JSON data
        benchmark_data = {
            "test_info": test_info,
            "all_results": all_results
        }

        # Save JSON
        json_file = results_dir / "benchmark_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved: {json_file.name}")

        # Generate visualization
        chart_file = results_dir / "throughput_chart.png"
        try:
            create_throughput_plot(benchmark_data, self.gpu_name, str(chart_file))
            print(f"  ✓ Saved: {chart_file.name}")
        except Exception as e:
            print(f"  ✗ Failed to generate chart: {e}")

        # Calculate optimal settings
        settings_file = results_dir / "optimal_settings.txt"
        try:
            best_config, elo_results = self.settings_calc.find_best_config(
                all_results, time_per_move, num_gpus
            )
            settings_output = self.settings_calc.format_results(
                best_config, elo_results, time_per_move, num_gpus
            )

            with open(settings_file, 'w', encoding='utf-8') as f:
                f.write(settings_output)
            print(f"  ✓ Saved: {settings_file.name}")

            # Print summary to console
            print("\n" + "-" * 80)
            print(f"Current Best Configuration (after batch size {current_batch}):")
            print(f"  Batch: {best_config['batch_size']}, Streams: {best_config['num_streams']}")
            print(f"  Search Threads: {best_config['num_search_threads']}")
            print(f"  ELO Effect: {best_config['elo_effect']:+.1f}")
            print("-" * 80)

        except Exception as e:
            print(f"  ✗ Failed to calculate optimal settings: {e}")


def detect_gpu_name() -> Optional[str]:
    """Try to detect GPU name using various methods."""
    # Method 1: Try pycuda
    try:
        import pycuda.driver as cuda
        cuda.init()
        if cuda.Device.count() > 0:
            return cuda.Device(0).name()
    except ImportError:
        pass

    # Method 2: Try nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader', '--id=0'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='TensorRT Benchmark Suite for KataGo',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--katago', required=True,
                       help='Path to KataGo executable')
    parser.add_argument('--model', required=True,
                       help='Path to model file (.bin.gz)')
    parser.add_argument('--trtexec', required=True,
                       help='Path to trtexec binary')
    parser.add_argument('--tensorrt-lib', required=True,
                       help='Path to TensorRT library directory')
    parser.add_argument('--cache-dir', required=True,
                       help='TensorRT cache directory')
    parser.add_argument('--gpu-name',
                       help='GPU name (auto-detect if not specified)')

    parser.add_argument('--batch-sizes', type=str, default='1-32',
                       help='Batch sizes to test (e.g., "1-32" or "4,8,16") (default: 1-32)')
    parser.add_argument('--stream-counts', type=str, default='1,2,3,4',
                       help='Stream counts to test (comma-separated) (default: 1,2,3,4)')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Benchmark duration per test in seconds (default: 5.0)')
    parser.add_argument('--warmup', type=int, default=500,
                       help='Warmup time in milliseconds (default: 500)')
    parser.add_argument('--time-per-move', type=float, default=3.0,
                       help='Time per move for ELO calculation (default: 3.0)')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='Number of GPUs for ELO calculation (default: 1)')

    args = parser.parse_args()

    # Parse batch sizes
    if '-' in args.batch_sizes:
        start, end = map(int, args.batch_sizes.split('-'))
        batch_sizes = list(range(start, end + 1))
    else:
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]

    # Parse stream counts
    stream_counts = [int(x.strip()) for x in args.stream_counts.split(',')]

    # Detect or use provided GPU name
    gpu_name = args.gpu_name
    if not gpu_name:
        gpu_name = detect_gpu_name()
        if gpu_name:
            print(f"Auto-detected GPU: {gpu_name}")
        else:
            print("Warning: Could not auto-detect GPU name, using 'Unknown GPU'")
            gpu_name = "Unknown GPU"

    # Create benchmark suite
    suite = TRTBenchmarkSuite(
        katago_executable=args.katago,
        model_file=args.model,
        trtexec_path=args.trtexec,
        tensorrt_lib_path=args.tensorrt_lib,
        cache_dir=args.cache_dir,
        gpu_name=gpu_name
    )

    # Run benchmark
    results_dir = suite.run_benchmark(
        batch_sizes=batch_sizes,
        stream_counts=stream_counts,
        duration=args.duration,
        warmup=args.warmup,
        time_per_move=args.time_per_move,
        num_gpus=args.num_gpus
    )
