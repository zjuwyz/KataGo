#!/usr/bin/env python3
"""
TensorRT Execution Runner

Wrapper for trtexec to benchmark TensorRT engine files.
"""

import os
import subprocess
import re
from typing import Dict, List, Tuple, Optional


class TRTExecRunner:
    """Wrapper for NVIDIA TensorRT trtexec tool"""

    def __init__(self, trtexec_path: str, tensorrt_lib_path: str):
        """
        Initialize TRTExec runner.

        Args:
            trtexec_path: Path to trtexec binary
            tensorrt_lib_path: Path to TensorRT libraries
        """
        self.trtexec_path = trtexec_path
        self.tensorrt_lib_path = tensorrt_lib_path

        # Validate paths
        if not os.path.exists(self.trtexec_path):
            raise FileNotFoundError(f"trtexec not found at {self.trtexec_path}")
        if not os.path.exists(self.tensorrt_lib_path):
            raise FileNotFoundError(f"TensorRT library path not found at {self.tensorrt_lib_path}")

    def _setup_environment(self) -> Dict[str, str]:
        """Set up environment variables for trtexec execution."""
        env = os.environ.copy()
        if "LD_LIBRARY_PATH" in env:
            env["LD_LIBRARY_PATH"] = f"{self.tensorrt_lib_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env["LD_LIBRARY_PATH"] = self.tensorrt_lib_path
        return env

    def extract_input_shapes(self, engine_path: str) -> Optional[List[Tuple[str, List[int]]]]:
        """
        Extract input tensor names and shapes from a TensorRT engine file.

        Args:
            engine_path: Path to the TensorRT engine file

        Returns:
            List of tuples (input_name, shape_list) or None if extraction fails
        """
        if not os.path.exists(engine_path):
            print(f"Error: Engine file does not exist: {engine_path}")
            return None

        try:
            cmd = [
                self.trtexec_path,
                f"--loadEngine={engine_path}",
                "--duration=0.1",  # Very short duration
                "--verbose"
            ]

            env = self._setup_environment()
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )

            if process.returncode != 0:
                print(f"Error: trtexec failed with return code {process.returncode}")
                print(f"stderr: {process.stderr}")
                return None

            output = process.stdout

            # Parse input shapes from the output
            # Look for lines like:
            # - "Set shape of input tensor InputMask to: 1x1x19x19" (dynamic shapes)
            # - "Input binding for InputMask with dimensions 1x1x19x19 is created." (fixed shapes)
            input_shapes = []
            for line in output.split('\n'):
                # Try dynamic shape pattern first
                if "Set shape of input tensor" in line and "to:" in line:
                    match = re.search(r'Set shape of input tensor (\w+) to: ([\dx]+)', line)
                    if match:
                        input_name = match.group(1)
                        shape_str = match.group(2)
                        dimensions = [int(x) for x in shape_str.split('x')]
                        input_shapes.append((input_name, dimensions))
                # Try fixed shape pattern
                elif "Input binding for" in line and "with dimensions" in line:
                    match = re.search(r'Input binding for (\w+) with dimensions ([\dx]+)', line)
                    if match:
                        input_name = match.group(1)
                        shape_str = match.group(2)
                        dimensions = [int(x) for x in shape_str.split('x')]
                        input_shapes.append((input_name, dimensions))

            if not input_shapes:
                print("Error: Could not parse input shapes from trtexec output")
                return None

            return input_shapes

        except subprocess.TimeoutExpired:
            print("Error: trtexec execution timed out while extracting input shapes")
            return None
        except Exception as e:
            print(f"Error extracting input shapes: {e}")
            return None

    def benchmark(
        self,
        engine_path: str,
        batch_size: int,
        num_streams: int = 2,
        duration: float = 5.0,
        warmup: int = 500,
        use_cuda_graph: bool = True,
        log_file: str = None
    ) -> Optional[Dict]:
        """
        Benchmark a TensorRT engine file using trtexec.

        Args:
            engine_path: Path to the TensorRT engine file
            batch_size: Batch size to use for inference
            num_streams: Number of CUDA streams (default: 2)
            duration: Duration in seconds to run benchmark (default: 5.0)
            warmup: Warmup time in milliseconds (default: 500)
            use_cuda_graph: Whether to use CUDA graph (default: True)
            log_file: Optional file path to save command and output

        Returns:
            Dictionary containing benchmark results:
            {
                "latency_ms": float,       # Average GPU latency in milliseconds
                "throughput_qps": float,   # Throughput in queries per second
                "input_shapes": List[Tuple[str, List[int]]],  # Input tensor shapes
                "trtexec_command": str,    # Command that was executed
                "trtexec_output": str      # Full output from trtexec
            }
            None if benchmarking fails
        """
        if not os.path.exists(engine_path):
            print(f"Error: Engine file does not exist: {engine_path}")
            return None

        print(f"Benchmarking engine: {engine_path}")
        print(f"  Batch size: {batch_size}, Streams: {num_streams}, Duration: {duration}s, "
              f"Warmup: {warmup}ms, CUDA Graph: {use_cuda_graph}")

        # Extract input shapes dynamically
        input_shapes_info = self.extract_input_shapes(engine_path)
        if not input_shapes_info:
            print("Error: Could not extract input shapes")
            return None

        print(f"Found {len(input_shapes_info)} input tensors:")
        for input_name, shape in input_shapes_info:
            print(f"  {input_name}: {shape}")

        try:
            # Build input shapes string with specified batch size
            input_shapes = []
            for input_name, shape in input_shapes_info:
                if len(shape) > 0:
                    # Replace batch dimension (first dimension) with requested batch size
                    new_shape = [batch_size] + shape[1:]
                    shape_str = "x".join(map(str, new_shape))
                    input_shapes.append(f"{input_name}:{shape_str}")
                else:
                    print(f"Warning: Input {input_name} has empty shape, skipping")

            if not input_shapes:
                print("Error: No valid input shapes found")
                return None

            input_shapes_str = ",".join(input_shapes)
            print(f"Using input shapes: {input_shapes_str}")

            # Build trtexec command
            cmd = [
                self.trtexec_path,
                f"--loadEngine={engine_path}",
                f"--shapes={input_shapes_str}",
                f"--duration={duration}",
                f"--warmUp={warmup}",
                f"--infStreams={num_streams}",
            ]

            # Enable multithreading when using multiple streams
            if num_streams > 1:
                cmd.append("--threads")

            if use_cuda_graph:
                cmd.append("--useCudaGraph")

            # Build command string for logging
            cmd_str = " ".join(cmd)

            # Run trtexec
            env = self._setup_environment()
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=int(duration + 30),  # Add buffer time
                env=env
            )

            output = process.stdout
            stderr = process.stderr

            # Log command and output if requested
            if log_file:
                try:
                    with open(log_file, 'w', encoding='utf-8') as f:
                        f.write("=" * 80 + "\n")
                        f.write("TensorRT Benchmark Log\n")
                        f.write("=" * 80 + "\n\n")
                        f.write("Command:\n")
                        f.write(cmd_str + "\n\n")
                        f.write("Environment:\n")
                        f.write(f"LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH', '')}\n\n")
                        f.write("=" * 80 + "\n")
                        f.write("Standard Output:\n")
                        f.write("=" * 80 + "\n")
                        f.write(output)
                        if stderr:
                            f.write("\n" + "=" * 80 + "\n")
                            f.write("Standard Error:\n")
                            f.write("=" * 80 + "\n")
                            f.write(stderr)
                        f.write("\n" + "=" * 80 + "\n")
                        f.write(f"Return Code: {process.returncode}\n")
                        f.write("=" * 80 + "\n")
                except Exception as e:
                    print(f"Warning: Could not write log file {log_file}: {e}")

            if process.returncode != 0:
                print(f"Error: trtexec failed with return code {process.returncode}")
                print(f"stderr: {stderr}")
                return None

            # Parse performance metrics from output
            # Look for: "GPU Compute Time: ... mean = X.XXX ms ..."
            gpu_latency_match = re.search(r'GPU Compute Time:.*?mean = ([\d.]+) ms', output)
            # Look for: "Throughput: XXX.XXX qps"
            throughput_match = re.search(r'Throughput: ([\d.]+) qps', output)

            if not gpu_latency_match or not throughput_match:
                print("Error: Could not parse performance metrics from trtexec output")
                print("Output sample:")
                print(output[-1000:] if len(output) > 1000 else output)
                return None

            latency_ms = float(gpu_latency_match.group(1))
            throughput_qps = float(throughput_match.group(1))

            result = {
                "latency_ms": latency_ms,
                "throughput_qps": throughput_qps,
                "input_shapes": input_shapes_info,
                "trtexec_command": cmd_str,
                "trtexec_output": output
            }

            print(f"Results: Latency = {latency_ms:.3f} ms, Throughput = {throughput_qps:.1f} qps")
            return result

        except subprocess.TimeoutExpired:
            print("Error: trtexec execution timed out")
            return None
        except Exception as e:
            print(f"Error running trtexec: {e}")
            return None


if __name__ == "__main__":
    # Test the runner
    runner = TRTExecRunner()

    # Test with a sample engine file
    test_engine = os.path.expanduser(
        "~/.katago/trtcache/trt-101303_gpu-426b8a57_net-kata1-b28c512nbt-s11623142656-d5506133042_7_exact19x19_batch8_fp16"
    )

    if os.path.exists(test_engine):
        print("Testing TRTExecRunner with sample engine file...\n")
        result = runner.benchmark(
            engine_path=test_engine,
            batch_size=8,
            num_streams=2,
            duration=2.0
        )

        if result:
            print("\nBenchmark successful!")
            print(f"Latency: {result['latency_ms']:.3f} ms")
            print(f"Throughput: {result['throughput_qps']:.1f} qps")
        else:
            print("\nBenchmark failed!")
    else:
        print(f"Test engine file not found: {test_engine}")
