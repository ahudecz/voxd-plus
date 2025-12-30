"""
GPU detection utilities for voxd-plus.

Provides functions to detect CUDA availability and GPU information
for optimizing whisper.cpp transcription.
"""

import subprocess
import shutil
import os
from pathlib import Path
from typing import Optional, Dict, Any


def detect_cuda() -> bool:
    """
    Check if CUDA is available on the system.

    Returns True if:
    - nvidia-smi is available and returns successfully
    - CUDA toolkit appears to be installed
    """
    # Check for nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return False

    try:
        result = subprocess.run(
            [nvidia_smi, "-L"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return False


def detect_cuda_toolkit() -> Optional[str]:
    """
    Detect CUDA toolkit installation and return version if found.

    Returns the CUDA version string (e.g., "12.4") or None if not found.
    """
    # Check nvcc (CUDA compiler)
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            result = subprocess.run(
                [nvcc, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse version from output like "release 12.4"
                for line in result.stdout.split('\n'):
                    if "release" in line.lower():
                        parts = line.split("release")
                        if len(parts) > 1:
                            version = parts[1].strip().split(",")[0].strip()
                            return version
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    # Check common CUDA paths
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-11",
        "/opt/cuda",
    ]

    for cuda_path in cuda_paths:
        version_file = Path(cuda_path) / "version.txt"
        if version_file.exists():
            try:
                content = version_file.read_text()
                # Parse "CUDA Version X.Y.Z"
                if "CUDA Version" in content:
                    version = content.split("CUDA Version")[1].strip().split()[0]
                    return version
            except Exception:
                pass

    return None


def get_gpu_info() -> Dict[str, Any]:
    """
    Get detailed GPU information.

    Returns a dict with:
    - available: bool - whether GPU is available
    - cuda_available: bool - whether CUDA is available
    - cuda_version: str | None - CUDA toolkit version
    - gpus: list of dicts with GPU details
    """
    info: Dict[str, Any] = {
        "available": False,
        "cuda_available": False,
        "cuda_version": None,
        "gpus": []
    }

    # Check CUDA availability
    info["cuda_available"] = detect_cuda()
    info["cuda_version"] = detect_cuda_toolkit()

    if not info["cuda_available"]:
        return info

    info["available"] = True

    # Get GPU list from nvidia-smi
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            # Get GPU names
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=index,name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:
                            gpu = {
                                "index": int(parts[0]) if parts[0].isdigit() else 0,
                                "name": parts[1],
                                "memory_mb": int(parts[2]) if parts[2].isdigit() else 0,
                                "compute_capability": parts[3] if len(parts) > 3 else "unknown"
                            }
                            info["gpus"].append(gpu)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError):
            pass

    return info


def get_whisper_device_flag(cfg=None) -> str:
    """
    Determine the device flag to pass to whisper-cli.

    Checks config for gpu_enabled and gpu_device settings.
    Returns "cuda" if GPU should be used, "cpu" otherwise.
    """
    # Check config settings
    if cfg:
        gpu_enabled = cfg.data.get("gpu_enabled", True)
        if not gpu_enabled:
            return "cpu"

        gpu_device = cfg.data.get("gpu_device", "auto")
        if gpu_device == "cpu":
            return "cpu"
        elif gpu_device == "cuda":
            return "cuda"
        # else: "auto" - fall through to detection

    # Auto-detect
    if detect_cuda():
        return "cuda"

    return "cpu"


def check_whisper_gpu_support(binary_path: str) -> bool:
    """
    Check if the whisper binary was built with GPU support.

    Returns True if the binary appears to support GPU acceleration.
    """
    if not binary_path or not Path(binary_path).exists():
        return False

    try:
        result = subprocess.run(
            [binary_path, "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Check if help output mentions GPU/CUDA options
        output = result.stdout + result.stderr
        if any(term in output.lower() for term in ["cuda", "gpu", "--device"]):
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return False


def should_rebuild_whisper(binary_path: str = None) -> bool:
    """
    Determine if whisper.cpp should be rebuilt with GPU support.

    Returns True if:
    - CUDA is available on the system
    - The current binary doesn't have GPU support
    """
    if not detect_cuda():
        return False

    if binary_path and check_whisper_gpu_support(binary_path):
        return False

    return True


def print_gpu_status():
    """Print GPU detection status for debugging."""
    info = get_gpu_info()

    print("\n[GPU Detection Status]")
    print(f"  CUDA Available: {info['cuda_available']}")
    print(f"  CUDA Version: {info['cuda_version'] or 'Not found'}")
    print(f"  GPUs Detected: {len(info['gpus'])}")

    for gpu in info["gpus"]:
        print(f"    [{gpu['index']}] {gpu['name']} ({gpu['memory_mb']} MB)")
        print(f"        Compute Capability: {gpu['compute_capability']}")

    if not info["gpus"]:
        print("    No NVIDIA GPUs detected")

    print()
