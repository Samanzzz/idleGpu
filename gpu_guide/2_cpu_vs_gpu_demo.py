#!/usr/bin/env python3
"""
CPU vs GPU Performance Comparison Demo for IDLE
===============================================

This script demonstrates the performance difference between CPU and GPU operations
using NumPy (CPU) and CuPy (GPU). It's designed to run in IDLE with clear output
and timing comparisons.

Required packages (install with pip):
    pip install numpy cupy-cuda12x  # or cupy-cuda11x for CUDA 11.x
    pip install memory-profiler

Author: Python GPU Guide
"""

import time
import sys
import traceback
from typing import Tuple, Optional, Dict, Any
import gc

def check_dependencies():
    """Check for required libraries and provide installation guidance."""
    missing_libs = []
    available_libs = {}
    
    # Check NumPy
    try:
        import numpy as np
        available_libs['numpy'] = np
        print("✓ NumPy available")
    except ImportError:
        missing_libs.append("numpy")
        print("✗ NumPy not found")
    
    # Check CuPy
    try:
        import cupy as cp
        available_libs['cupy'] = cp
        print("✓ CuPy available")
        
        # Test CUDA availability
        try:
            cp.cuda.Device(0).compute_capability
            print("✓ CUDA GPU detected and accessible")
        except Exception as e:
            print(f"⚠ CuPy installed but CUDA issue: {e}")
            
    except ImportError:
        missing_libs.append("cupy-cuda12x")  # or cupy-cuda11x
        print("✗ CuPy not found")
    
    # Check memory profiler (optional)
    try:
        import memory_profiler
        available_libs['memory_profiler'] = memory_profiler
        print("✓ memory-profiler available")
    except ImportError:
        print("⚠ memory-profiler not found (optional)")
    
    if missing_libs:
        print(f"\nMissing critical libraries: {', '.join(missing_libs)}")
        print("Install with:")
        for lib in missing_libs:
            if 'cupy' in lib:
                print(f"  pip install {lib}  # Choose cuda11x or cuda12x based on your CUDA version")
            else:
                print(f"  pip install {lib}")
        print("\nThen restart IDLE and run this script again.")
        return available_libs, False
    
    return available_libs, True

class PerformanceTimer:
    """Simple context manager for timing operations."""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        gc.collect()  # Clean up memory before timing
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"{self.operation_name}: {self.duration:.4f} seconds")

def get_memory_usage():
    """Get current memory usage (if memory_profiler is available)."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except:
        return None

def matrix_multiplication_demo(size: int = 2000):
    """Compare CPU vs GPU matrix multiplication performance."""
    print(f"\n{'='*60}")
    print(f"MATRIX MULTIPLICATION DEMO (Size: {size}x{size})")
    print(f"{'='*60}")
    
    try:
        import numpy as np
        import cupy as cp
        
        print(f"Creating random matrices of size {size}x{size}...")
        
        # Create CPU matrices
        with PerformanceTimer("CPU matrix creation"):
            cpu_a = np.random.random((size, size)).astype(np.float32)
            cpu_b = np.random.random((size, size)).astype(np.float32)
        
        # Create GPU matrices
        with PerformanceTimer("GPU matrix creation and transfer"):
            gpu_a = cp.random.random((size, size), dtype=cp.float32)
            gpu_b = cp.random.random((size, size), dtype=cp.float32)
        
        print("\nPerforming matrix multiplication...")
        
        # CPU computation
        with PerformanceTimer("CPU matrix multiplication"):
            cpu_result = np.dot(cpu_a, cpu_b)
        
        # GPU computation
        with PerformanceTimer("GPU matrix multiplication"):
            gpu_result = cp.dot(gpu_a, gpu_b)
        
        # Transfer result back to CPU for verification
        with PerformanceTimer("GPU result transfer to CPU"):
            gpu_result_cpu = cp.asnumpy(gpu_result)
        
        # Verify results are similar (accounting for floating point differences)
        max_diff = np.max(np.abs(cpu_result - gpu_result_cpu))
        print(f"\nVerification - Max difference between CPU and GPU results: {max_diff:.2e}")
        
        if max_diff < 1e-5:
            print("✓ Results match (within floating point precision)")
        else:
            print("⚠ Results differ significantly - check your setup")
        
        # Memory usage
        mem_usage = get_memory_usage()
        if mem_usage:
            print(f"Current memory usage: {mem_usage:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error in matrix multiplication demo: {e}")
        traceback.print_exc()
        return False

def element_wise_operations_demo(size: int = 10000000):
    """Compare CPU vs GPU element-wise operations."""
    print(f"\n{'='*60}")
    print(f"ELEMENT-WISE OPERATIONS DEMO (Size: {size:,} elements)")
    print(f"{'='*60}")
    
    try:
        import numpy as np
        import cupy as cp
        
        print(f"Creating arrays with {size:,} elements...")
        
        # Create CPU arrays
        with PerformanceTimer("CPU array creation"):
            cpu_x = np.random.random(size).astype(np.float32)
            cpu_y = np.random.random(size).astype(np.float32)
        
        # Create GPU arrays
        with PerformanceTimer("GPU array creation"):
            gpu_x = cp.random.random(size, dtype=cp.float32)
            gpu_y = cp.random.random(size, dtype=cp.float32)
        
        print("\nPerforming element-wise operations (sin(x) + cos(y) * exp(x/10))...")
        
        # CPU computation
        with PerformanceTimer("CPU element-wise operations"):
            cpu_result = np.sin(cpu_x) + np.cos(cpu_y) * np.exp(cpu_x / 10)
        
        # GPU computation
        with PerformanceTimer("GPU element-wise operations"):
            gpu_result = cp.sin(gpu_x) + cp.cos(gpu_y) * cp.exp(gpu_x / 10)
        
        # Transfer result back for verification
        with PerformanceTimer("GPU result transfer"):
            gpu_result_cpu = cp.asnumpy(gpu_result)
        
        # Verify results
        max_diff = np.max(np.abs(cpu_result - gpu_result_cpu))
        print(f"\nVerification - Max difference: {max_diff:.2e}")
        
        if max_diff < 1e-6:
            print("✓ Results match")
        else:
            print("⚠ Results differ - check setup")
        
        return True
        
    except Exception as e:
        print(f"Error in element-wise operations demo: {e}")
        traceback.print_exc()
        return False

def reduction_operations_demo(size: int = 50000000):
    """Compare CPU vs GPU reduction operations (sum, mean, etc.)."""
    print(f"\n{'='*60}")
    print(f"REDUCTION OPERATIONS DEMO (Size: {size:,} elements)")
    print(f"{'='*60}")
    
    try:
        import numpy as np
        import cupy as cp
        
        print(f"Creating array with {size:,} elements...")
        
        # Create arrays
        with PerformanceTimer("CPU array creation"):
            cpu_data = np.random.random(size).astype(np.float32)
        
        with PerformanceTimer("GPU array creation"):
            gpu_data = cp.random.random(size, dtype=cp.float32)
        
        print("\nPerforming reduction operations...")
        
        # CPU reductions
        with PerformanceTimer("CPU sum"):
            cpu_sum = np.sum(cpu_data)
        
        with PerformanceTimer("CPU mean"):
            cpu_mean = np.mean(cpu_data)
        
        with PerformanceTimer("CPU std"):
            cpu_std = np.std(cpu_data)
        
        # GPU reductions
        with PerformanceTimer("GPU sum"):
            gpu_sum = cp.sum(gpu_data)
        
        with PerformanceTimer("GPU mean"):
            gpu_mean = cp.mean(gpu_data)
        
        with PerformanceTimer("GPU std"):
            gpu_std = cp.std(gpu_data)
        
        # Convert GPU results to CPU for comparison
        gpu_sum_cpu = float(gpu_sum)
        gpu_mean_cpu = float(gpu_mean)
        gpu_std_cpu = float(gpu_std)
        
        print(f"\nResults comparison:")
        print(f"Sum  - CPU: {cpu_sum:.6f}, GPU: {gpu_sum_cpu:.6f}, Diff: {abs(cpu_sum - gpu_sum_cpu):.2e}")
        print(f"Mean - CPU: {cpu_mean:.6f}, GPU: {gpu_mean_cpu:.6f}, Diff: {abs(cpu_mean - gpu_mean_cpu):.2e}")
        print(f"Std  - CPU: {cpu_std:.6f}, GPU: {gpu_std_cpu:.6f}, Diff: {abs(cpu_std - gpu_std_cpu):.2e}")
        
        return True
        
    except Exception as e:
        print(f"Error in reduction operations demo: {e}")
        traceback.print_exc()
        return False

def memory_transfer_analysis():
    """Analyze the cost of memory transfers between CPU and GPU."""
    print(f"\n{'='*60}")
    print("MEMORY TRANSFER ANALYSIS")
    print(f"{'='*60}")
    
    try:
        import numpy as np
        import cupy as cp
        
        sizes = [1000, 10000, 100000, 1000000, 10000000]
        
        print("Analyzing memory transfer costs for different array sizes...")
        print(f"{'Size':>10} {'CPU->GPU (ms)':>15} {'GPU->CPU (ms)':>15} {'Data (MB)':>12}")
        print("-" * 60)
        
        for size in sizes:
            # Create CPU array
            cpu_array = np.random.random(size).astype(np.float32)
            data_size_mb = cpu_array.nbytes / 1024 / 1024
            
            # Measure CPU to GPU transfer
            start_time = time.perf_counter()
            gpu_array = cp.asarray(cpu_array)
            cpu_to_gpu_time = (time.perf_counter() - start_time) * 1000
            
            # Measure GPU to CPU transfer
            start_time = time.perf_counter()
            result_cpu = cp.asnumpy(gpu_array)
            gpu_to_cpu_time = (time.perf_counter() - start_time) * 1000
            
            print(f"{size:>10,} {cpu_to_gpu_time:>12.2f} {gpu_to_cpu_time:>12.2f} {data_size_mb:>9.2f}")
        
        print("\nKey insights:")
        print("- Memory transfers have overhead, especially for small arrays")
        print("- GPU acceleration is most beneficial for large datasets")
        print("- Consider keeping data on GPU between operations when possible")
        
        return True
        
    except Exception as e:
        print(f"Error in memory transfer analysis: {e}")
        traceback.print_exc()
        return False

def gpu_device_info():
    """Display detailed GPU device information."""
    print(f"\n{'='*60}")
    print("GPU DEVICE INFORMATION")
    print(f"{'='*60}")
    
    try:
        import cupy as cp
        
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        
        print(f"Device ID: {device.id}")
        print(f"Device Name: {props['name'].decode()}")
        print(f"Compute Capability: {device.compute_capability}")
        print(f"Total Memory: {props['totalGlobalMem'] / 1024**3:.2f} GB")
        print(f"Multiprocessors: {props['multiProcessorCount']}")
        print(f"Max Threads per Block: {props['maxThreadsPerBlock']}")
        print(f"Max Block Dimensions: {props['maxThreadsDim']}")
        print(f"Max Grid Dimensions: {props['maxGridSize']}")
        print(f"Warp Size: {props['warpSize']}")
        
        # Memory info
        mempool = cp.get_default_memory_pool()
        print(f"\nMemory Pool Info:")
        print(f"Used bytes: {mempool.used_bytes() / 1024**2:.2f} MB")
        print(f"Total bytes: {mempool.total_bytes() / 1024**2:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error getting GPU device info: {e}")
        return False

def performance_summary(results: Dict[str, bool]):
    """Display a summary of all performance tests."""
    print(f"\n{'='*60}")
    print("PERFORMANCE TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"Tests completed: {passed_tests}/{total_tests}")
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests completed successfully!")
        print("\nKey takeaways:")
        print("- GPU acceleration provides significant speedup for large datasets")
        print("- Memory transfer overhead affects small operations")
        print("- Element-wise and matrix operations benefit most from GPU acceleration")
        print("- Consider data size and operation complexity when choosing CPU vs GPU")
    else:
        print(f"\n⚠ {total_tests - passed_tests} test(s) failed.")
        print("This might indicate:")
        print("- CUDA/GPU driver issues")
        print("- Insufficient GPU memory")
        print("- Library installation problems")

def main():
    """Main function to run all CPU vs GPU demonstrations."""
    print("CPU vs GPU Performance Comparison Demo")
    print("=" * 50)
    
    # Check dependencies
    print("\nChecking dependencies...")
    libs, all_available = check_dependencies()
    
    if not all_available:
        print("\nCannot proceed without required libraries.")
        return
    
    # Display GPU info if available
    gpu_info_success = gpu_device_info()
    
    # Run performance tests
    results = {}
    
    print(f"\n{'='*60}")
    print("STARTING PERFORMANCE COMPARISONS")
    print("(This may take a few minutes...)")
    print(f"{'='*60}")
    
    # Matrix multiplication test
    results["Matrix Multiplication"] = matrix_multiplication_demo(size=2000)
    
    # Element-wise operations test
    results["Element-wise Operations"] = element_wise_operations_demo(size=10000000)
    
    # Reduction operations test
    results["Reduction Operations"] = reduction_operations_demo(size=50000000)
    
    # Memory transfer analysis
    results["Memory Transfer Analysis"] = memory_transfer_analysis()
    
    # Display summary
    performance_summary(results)
    
    print(f"\n{'='*60}")
    print("NEXT STEPS:")
    print("1. Run '3_optimization_examples.py' for advanced GPU optimization techniques")
    print("2. Experiment with different array sizes to find optimal GPU usage")
    print("3. Monitor GPU usage with '1_gpu_info.py' while running computations")
    print("4. Consider using CuPy in your own projects for numerical computations")

if __name__ == "__main__":
    main()
