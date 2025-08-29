#!/usr/bin/env python3
"""
GPU Device Checker
This script checks for available GPU devices and provides detailed information.
"""

import platform
import subprocess
import sys

import torch


def check_cuda_availability():
    """Check CUDA availability and provide detailed information."""
    print("=" * 50)
    print("CUDA GPU CHECK")
    print("=" * 50)

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        # Get CUDA version
        try:
            # Try different ways to get CUDA version
            if hasattr(torch, "version") and hasattr(torch.version, "cuda"):
                cuda_version = torch.version.cuda
            else:
                # Alternative method using runtime API
                cuda_version = (
                    torch.version.cuda if hasattr(torch.version, "cuda") else "Unknown"
                )
            print(f"CUDA Version: {cuda_version}")
        except Exception:
            print("CUDA Version: Unable to determine")

        # Get number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")

        # Get current GPU
        current_device = torch.cuda.current_device()
        print(f"Current GPU Device: {current_device}")

        # Get GPU details for each device
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i} Details:")
            print(f"  Name: {props.name}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Multi-processor Count: {props.multi_processor_count}")
            print(f"  CUDA Capability: {props.major}.{props.minor}")

            # Memory usage
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Memory Allocated: {memory_allocated:.2f} GB")
            print(f"  Memory Reserved: {memory_reserved:.2f} GB")
    else:
        print("No CUDA GPUs available")

        # Check if CUDA toolkit is installed
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                print("CUDA toolkit is installed but no compatible GPU found")
                print(result.stdout)
        except FileNotFoundError:
            print("CUDA toolkit not found in PATH")


def check_mps_availability():
    """Check MPS (Metal Performance Shaders) availability for Apple Silicon."""
    print("\n" + "=" * 50)
    print("MPS (Apple Silicon) CHECK")
    print("=" * 50)

    # Check if MPS is available (PyTorch 1.12+)
    if hasattr(torch.backends, "mps"):
        mps_available = torch.backends.mps.is_available()
        print(f"MPS Available: {mps_available}")

        if mps_available:
            mps_built = torch.backends.mps.is_built()
            print(f"MPS Built: {mps_built}")

            if mps_built:
                print("Apple Silicon GPU is available for acceleration")
            else:
                print("MPS not properly built in this PyTorch installation")
        else:
            print("MPS not available (not on Apple Silicon or not supported)")
    else:
        print("MPS backend not available in this PyTorch version")


def check_system_info():
    """Display system information."""
    print("\n" + "=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)

    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Machine: {platform.machine()}")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")


def test_gpu_computation():
    """Test GPU computation with a simple tensor operation."""
    print("\n" + "=" * 50)
    print("GPU COMPUTATION TEST")
    print("=" * 50)

    # Test CUDA
    if torch.cuda.is_available():
        print("Testing CUDA computation...")
        try:
            # Create tensors on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()

            # Perform computation
            z = torch.matmul(x, y)

            print("✓ CUDA computation successful")
            print(f"  Result tensor shape: {z.shape}")
            print(f"  Result tensor device: {z.device}")
        except Exception as e:
            print(f"✗ CUDA computation failed: {e}")

    # Test MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Testing MPS computation...")
        try:
            # Create tensors on MPS
            x = torch.randn(1000, 1000).to("mps")
            y = torch.randn(1000, 1000).to("mps")

            # Perform computation
            z = torch.matmul(x, y)

            print("✓ MPS computation successful")
            print(f"  Result tensor shape: {z.shape}")
            print(f"  Result tensor device: {z.device}")
        except Exception as e:
            print(f"✗ MPS computation failed: {e}")


def get_recommended_device():
    """Get the recommended device for PyTorch operations."""
    print("\n" + "=" * 50)
    print("RECOMMENDED DEVICE")
    print("=" * 50)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Recommended device: {device}")
        print("Using CUDA GPU for acceleration")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Recommended device: {device}")
        print("Using Apple Silicon GPU for acceleration")
    else:
        device = torch.device("cpu")
        print(f"Recommended device: {device}")
        print("No GPU acceleration available, using CPU")

    return device


def main():
    """Main function to run all GPU checks."""
    print("GPU Device Checker")
    print(
        "This script will check for available GPU devices and test their functionality.\n"
    )

    # Run all checks
    check_system_info()
    check_cuda_availability()
    check_mps_availability()
    test_gpu_computation()
    recommended_device = get_recommended_device()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if torch.cuda.is_available():
        print("✓ CUDA GPU available")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("✓ Apple Silicon GPU available")
    else:
        print("⚠ No GPU acceleration available")

    print(f"Recommended PyTorch device: {recommended_device}")


if __name__ == "__main__":
    main()
