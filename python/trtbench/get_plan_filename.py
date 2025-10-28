#!/usr/bin/env python3
"""
TensorRT Plan Filename Generator

Generates TensorRT plan cache filenames based on KataGo's naming convention.
This replicates the logic from cpp/neuralnet/trtbackend.cpp without needing to
run KataGo.
"""

import hashlib
import os
import ctypes
from pathlib import Path
from typing import Optional


class PlanFilenameGenerator:
    """Generates TensorRT plan cache filenames following KataGo's convention"""

    # Bump this when between katago versions we want to forcibly drop old caches
    # From cpp/neuralnet/trtbackend.cpp ModelParser::tuneSalt
    TUNE_SALT = 7

    def __init__(
        self,
        tensorrt_lib_path: str,
        cache_dir: str
    ):
        """
        Initialize the plan filename generator.

        Args:
            tensorrt_lib_path: Path to TensorRT libnvinfer.so library
            cache_dir: TensorRT cache directory
        """
        self.tensorrt_lib_path = tensorrt_lib_path
        self.cache_dir = cache_dir
        self._trt_version = None

    def get_tensorrt_version(self) -> int:
        """
        Get TensorRT library version by calling getInferLibVersion().

        Returns:
            TensorRT version as integer (e.g., 101303 for 10.13.3)
        """
        if self._trt_version is not None:
            return self._trt_version

        try:
            lib = ctypes.CDLL(self.tensorrt_lib_path)
            self._trt_version = lib.getInferLibVersion()
            return self._trt_version
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TensorRT library at {self.tensorrt_lib_path}: {e}"
            )

    def get_gpu_device_hash(self, gpu_name: str) -> str:
        """
        Compute GPU device identifier hash.

        This replicates the logic from trtbackend.cpp:
        - SHA256 hash of GPU name
        - Truncated to first 4 bytes
        - Formatted as 8-character hex string

        Args:
            gpu_name: GPU device name (e.g., "NVIDIA GeForce RTX 5080")

        Returns:
            8-character hex string (e.g., "426b8a57")
        """
        # SHA256 hash of device name
        device_hash = hashlib.sha256(gpu_name.encode('utf-8')).digest()

        # Truncate to first 4 bytes and format as hex
        return ''.join(f'{b:02x}' for b in device_hash[:4])

    def generate_plan_filename(
        self,
        model_name: str,
        gpu_name: str,
        batch_size: int,
        nn_x_len: int = 19,
        nn_y_len: int = 19,
        require_exact_nn_len: bool = True,
        use_fp16: bool = True,
        use_cache_plan: bool = True
    ) -> str:
        """
        Generate TensorRT plan cache filename.

        This replicates the filename generation logic from trtbackend.cpp.

        Args:
            model_name: Model name (e.g., "kata1-b28c512nbt-s11623142656-d5506133042")
            gpu_name: GPU device name (e.g., "NVIDIA GeForce RTX 5080")
            batch_size: Batch size for the plan
            nn_x_len: Neural network X dimension (default: 19)
            nn_y_len: Neural network Y dimension (default: 19)
            require_exact_nn_len: Whether to require exact board size (default: True)
            use_fp16: Whether to use FP16 precision (default: True)
            use_cache_plan: Whether CACHE_TENSORRT_PLAN was enabled during build (default: True)

        Returns:
            Full path to the plan cache file
        """
        trt_version = self.get_tensorrt_version()
        device_ident = self.get_gpu_device_hash(gpu_name)
        exact_or_max = "exact" if require_exact_nn_len else "max"
        fp_bits = 16 if use_fp16 else 32

        if use_cache_plan:
            # Format: trt-{version}_gpu-{hash}_net-{name}_{salt}_{exact/max}{Y}x{X}_batch{N}_fp{bits}
            filename = (
                f"trt-{trt_version}_gpu-{device_ident}_net-{model_name}_"
                f"{self.TUNE_SALT}_{exact_or_max}{nn_y_len}x{nn_x_len}_"
                f"batch{batch_size}_fp{fp_bits}"
            )
        else:
            # When CACHE_TENSORRT_PLAN is not defined, a different naming scheme might be used
            # For now, we assume the same format (this may need adjustment based on actual behavior)
            filename = (
                f"trt-{trt_version}_gpu-{device_ident}_net-{model_name}_"
                f"{self.TUNE_SALT}_{exact_or_max}{nn_y_len}x{nn_x_len}_"
                f"batch{batch_size}_fp{fp_bits}"
            )

        return os.path.join(self.cache_dir, filename)

    def generate_timing_cache_filename(
        self,
        tune_hash: bytes,
        gpu_name: str,
        batch_size: int,
        nn_x_len: int = 19,
        nn_y_len: int = 19,
        require_exact_nn_len: bool = True,
        use_fp16: bool = True
    ) -> str:
        """
        Generate TensorRT timing cache filename.

        Args:
            tune_hash: Model tune hash (raw bytes, typically first 6 bytes are used)
            gpu_name: GPU device name
            batch_size: Batch size
            nn_x_len: Neural network X dimension (default: 19)
            nn_y_len: Neural network Y dimension (default: 19)
            require_exact_nn_len: Whether to require exact board size (default: True)
            use_fp16: Whether to use FP16 precision (default: True)

        Returns:
            Full path to the timing cache file
        """
        trt_version = self.get_tensorrt_version()
        device_ident = self.get_gpu_device_hash(gpu_name)

        # Truncate to 6 bytes and format as hex
        tune_ident = ''.join(f'{b:02x}' for b in tune_hash[:6])

        exact_or_max = "exact" if require_exact_nn_len else "max"
        fp_bits = 16 if use_fp16 else 32

        # Format: trt-{version}_gpu-{hash}_tune-{tunehash}_{exact/max}{Y}x{X}_batch{N}_fp{bits}
        filename = (
            f"trt-{trt_version}_gpu-{device_ident}_tune-{tune_ident}_"
            f"{exact_or_max}{nn_y_len}x{nn_x_len}_batch{batch_size}_fp{fp_bits}"
        )

        return os.path.join(self.cache_dir, filename)


def get_gpu_name_from_cuda() -> Optional[str]:
    """
    Get GPU name using pycuda (if available).

    Returns:
        GPU name string, or None if pycuda is not available
    """
    try:
        import pycuda.driver as cuda
        cuda.init()
        if cuda.Device.count() == 0:
            return None
        device = cuda.Device(0)
        return device.name()
    except ImportError:
        return None


if __name__ == "__main__":
    # Example usage
    generator = PlanFilenameGenerator()

    # Example parameters (matching the test file we used earlier)
    model_name = "kata1-b28c512nbt-s11623142656-d5506133042"
    gpu_name = "NVIDIA GeForce RTX 5080"
    batch_size = 8

    # Try to auto-detect GPU name
    detected_gpu = get_gpu_name_from_cuda()
    if detected_gpu:
        print(f"Detected GPU: {detected_gpu}")
        gpu_name = detected_gpu

    print(f"\nGenerating plan filename for:")
    print(f"  Model: {model_name}")
    print(f"  GPU: {gpu_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Board size: 19x19")
    print(f"  FP16: True")
    print()

    # Generate plan filename
    plan_file = generator.generate_plan_filename(
        model_name=model_name,
        gpu_name=gpu_name,
        batch_size=batch_size
    )

    print(f"Generated plan filename:")
    print(f"  {plan_file}")
    print()

    # Check if file exists
    if os.path.exists(plan_file):
        print(f"✓ Plan file exists!")
        file_size = os.path.getsize(plan_file) / (1024 * 1024)
        print(f"  Size: {file_size:.1f} MB")
    else:
        print(f"✗ Plan file does not exist")

    # Generate device hash
    device_hash = generator.get_gpu_device_hash(gpu_name)
    print(f"\nGPU device hash: {device_hash}")
    print(f"TensorRT version: {generator.get_tensorrt_version()}")
