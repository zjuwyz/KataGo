#!/usr/bin/env python3
"""
TensorRT Plan Cache Generator

Generates TensorRT plan cache files by launching KataGo once per batch size.
This is much faster than running full benchmarks to generate plans.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional


class PlanGenerator:
    """Generator for TensorRT plan cache files"""

    def __init__(
        self,
        katago_executable: str,
        model_file: str,
        tensorrt_lib_path: str,
        cache_dir: str
    ):
        """
        Initialize plan generator.

        Args:
            katago_executable: Path to katago binary
            model_file: Path to model file (.bin.gz)
            tensorrt_lib_path: Path to TensorRT libraries
            cache_dir: TensorRT cache directory
        """
        self.katago_executable = Path(katago_executable)
        self.model_file = Path(model_file)
        self.tensorrt_lib_path = Path(tensorrt_lib_path)
        self.cache_dir = Path(cache_dir)

        # Validate paths
        if not self.katago_executable.exists():
            raise FileNotFoundError(f"KataGo executable not found: {self.katago_executable}")
        if not self.model_file.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_file}")
        if not self.tensorrt_lib_path.exists():
            raise FileNotFoundError(f"TensorRT library path not found: {self.tensorrt_lib_path}")

    def _setup_environment(self) -> dict:
        """Set up environment variables for KataGo execution."""
        env = os.environ.copy()
        lib_path = str(self.tensorrt_lib_path)
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{lib_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = lib_path
        return env

    def generate_plan(
        self,
        batch_size: int,
        timeout: int = 120
    ) -> bool:
        """
        Generate a TensorRT plan cache file for the specified batch size.

        This works by:
        1. Starting KataGo in GTP mode with the specified batch size
        2. Waiting for it to load the model and generate the plan
        3. Sending 'quit' command to cleanly shut down

        Args:
            batch_size: Batch size to generate plan for
            timeout: Maximum time to wait for plan generation (seconds)

        Returns:
            True if plan generation succeeded, False otherwise
        """
        print(f"Generating TensorRT plan for batch size {batch_size}...")

        try:
            # Use the example GTP config that comes with KataGo
            config_file = self.katago_executable.parent / "configs" / "gtp_example.cfg"
            if not config_file.exists():
                raise FileNotFoundError(f"GTP example config not found: {config_file}")

            cmd = [
                str(self.katago_executable),
                "gtp",
                "-model", str(self.model_file),
                "-config", str(config_file),
                "-override-config",
                f"nnMaxBatchSize={batch_size},"
                f"numSearchThreads=1"  # Minimal search threads, we just need the plan
            ]

            env = self._setup_environment()

            # Start KataGo process
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env
            )

            try:

                print(f"  Started KataGo process (PID: {process.pid})")

                # Read output until we see "GTP ready" or find the model name
                output_lines = []
                model_name = None
                start_time = time.time()

                while True:
                    # Check timeout
                    if time.time() - start_time > timeout:
                        print(f"  Error: Timeout after {timeout} seconds")
                        return None

                    # Read a line
                    line = process.stdout.readline()
                    if not line:
                        # Process terminated
                        break

                    output_lines.append(line.strip())

                    # Print important lines
                    if any(keyword in line for keyword in ["Model name:", "GTP ready", "loaded model"]):
                        print(f"  KataGo: {line.strip()}")

                    # Extract model name
                    if "Model name:" in line:
                        import re
                        match = re.search(r"Model name: (.+)", line)
                        if match:
                            model_name = match.group(1).strip()

                    # Check if ready
                    if "GTP ready" in line:
                        print(f"  KataGo is ready!")
                        break

                    # Safety check: stop if too many lines
                    if len(output_lines) > 500:
                        print("  Error: Too many output lines, something went wrong")
                        return None

                # Send quit command - KataGo will always exit cleanly on "quit"
                print("  Sending 'quit' command...")
                process.stdin.write("quit\n")
                process.stdin.flush()

                # Wait for clean shutdown
                process.wait(timeout=10)
                print(f"  KataGo exited cleanly (return code: {process.returncode})")

                if process.returncode == 0:
                    print(f"  ✓ Plan generation completed successfully")
                    return True
                else:
                    print(f"  ✗ KataGo exited with error code: {process.returncode}")
                    return False

            except subprocess.TimeoutExpired:
                print(f"  Error: KataGo did not exit within timeout")
                return False

        except Exception as e:
            print(f"  Error generating plan: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _detect_gpu_name(self, output_lines: list) -> Optional[str]:
        """Try to detect GPU name from KataGo output."""
        for line in output_lines:
            # Look for GPU device name in output
            # Common patterns: "Using GPU 0: NVIDIA GeForce RTX 5080"
            if "GPU" in line and "NVIDIA" in line:
                import re
                match = re.search(r'GPU \d+: (.+)', line)
                if match:
                    return match.group(1).strip()
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate TensorRT plan cache files for KataGo"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16",
        help="Comma-separated list of batch sizes to generate (default: 1,4,8,16)"
    )
    parser.add_argument(
        "--katago",
        type=str,
        help="Path to katago executable (default: auto-detect)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to model file (default: ~/.katago/weights/latest.bin.gz)"
    )

    args = parser.parse_args()

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print("TensorRT Plan Cache Generator")
    print("=" * 60)
    print(f"Batch sizes to generate: {batch_sizes}")
    print()

    # Create generator
    generator = PlanGenerator(
        katago_executable=args.katago,
        model_file=args.model
    )

    print(f"Using:")
    print(f"  KataGo: {generator.katago_executable}")
    print(f"  Model: {generator.model_file}")
    print(f"  TensorRT: {generator.tensorrt_lib_path}")
    print()

    # Generate plans
    successful = 0
    failed = 0

    for i, batch_size in enumerate(batch_sizes, 1):
        print(f"[{i}/{len(batch_sizes)}] Generating plan for batch size {batch_size}...")
        result = generator.generate_plan(batch_size)

        if result:
            successful += 1
        else:
            failed += 1

        print()

    print("=" * 60)
    print(f"Plan generation complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print("=" * 60)
