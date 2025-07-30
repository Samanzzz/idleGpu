#!/usr/bin/env python3
"""
Advanced GPU Optimization Examples for IDLE
===========================================

This script demonstrates advanced GPU optimization techniques including:
- Custom CUDA kernels with Numba
- Memory management strategies
- Batch processing optimization
- Data pipeline optimization

Required packages (install with pip):
    pip install numba cupy-cuda12x numpy

Author: Python GPU Guide
"""

import time
import sys
import traceback
from typing import List, Tuple, Optional, Dict, Any
import gc
import math

def check_dependencies():
    """Check for required libraries."""
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
    except ImportError:
        missing_libs.append("cupy-cuda12x")
        print("✗ CuPy not found")
    
    # Check Numba
    try:
        import numba
        from numba import cuda
        available_libs['numba'] = numba
        available_libs['cuda'] = cuda
        print("✓ Numba with CUDA support available")
        
        # Test CUDA availability
        try:
            cuda.detect()
            print("✓ CUDA devices detected by Numba")
        except Exception as e:
            print(f"⚠ Numba CUDA issue: {e}")
            
    except ImportError:
        missing_libs.append("numba")
        print("✗ Numba not found")
    
    if missing_libs:
        print(f"\nMissing libraries: {', '.join(missing_libs)}")
        print("Install with: pip install " + " ".join(missing_libs))
        return available_libs, False
    
    return available_libs, True

class GPUTimer:
    """GPU-aware timing context manager."""
    
    def __init__(self, name: str, use_gpu_sync: bool = True):
        self.name = name
        self.use_gpu_sync = use_gpu_sync
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        if self.use_gpu_sync:
            try:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
            except:
                pass
        
        gc.collect()
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_gpu_sync:
            try:
                import cupy as cp
                cp.cuda.Stream.null.synchronize()
            except:
                pass
        
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"{self.name}: {self.duration:.4f} seconds")

def numba_cuda_kernel_example():
    """Demonstrate custom CUDA kernels using Numba."""
    print(f"\n{'='*60}")
    print("NUMBA CUDA KERNEL EXAMPLE")
    print(f"{'='*60}")
    
    try:
        import numpy as np
        import numba
        from numba import cuda
        
        # Define a simple CUDA kernel for vector addition
        @cuda.jit
        def vector_add_kernel(a, b, c):
            """CUDA kernel for element-wise vector addition."""
            idx = cuda.grid(1)  # Get the thread index
            if idx < a.size:
                c[idx] = a[idx] + b[idx]
        
        # Define a more complex kernel for matrix multiplication
        @cuda.jit
        def matrix_multiply_kernel(A, B, C):
            """CUDA kernel for matrix multiplication."""
            row, col = cuda.grid(2)
            if row < C.shape[0] and col < C.shape[1]:
                tmp = 0.0
                for k in range(A.shape[1]):
                    tmp += A[row, k] * B[k, col]
                C[row, col] = tmp
        
        # Vector addition example
        print("Vector Addition with Custom CUDA Kernel:")
        size = 1000000
        
        # Create host arrays
        a_host = np.random.random(size).astype(np.float32)
        b_host = np.random.random(size).astype(np.float32)
        c_host = np.zeros(size, dtype=np.float32)
        
        # Allocate GPU memory
        with GPUTimer("GPU memory allocation"):
            a_gpu = cuda.to_device(a_host)
            b_gpu = cuda.to_device(b_host)
            c_gpu = cuda.device_array(size, dtype=np.float32)
        
        # Configure kernel launch parameters
        threads_per_block = 256
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
        
        print(f"Kernel configuration: {blocks_per_grid} blocks, {threads_per_block} threads per block")
        
        # Launch kernel
        with GPUTimer("Custom CUDA kernel execution"):
            vector_add_kernel[blocks_per_grid, threads_per_block](a_gpu, b_gpu, c_gpu)
        
        # Copy result back to host
        with GPUTimer("GPU to CPU transfer"):
            result = c_gpu.copy_to_host()
        
        # Verify result
        expected = a_host + b_host
        max_error = np.max(np.abs(result - expected))
        print(f"Max error: {max_error:.2e}")
        
        if max_error < 1e-6:
            print("✓ Custom kernel result verified")
        else:
            print("⚠ Custom kernel result verification failed")
        
        # Matrix multiplication example
        print(f"\nMatrix Multiplication with Custom CUDA Kernel:")
        matrix_size = 512
        
        A_host = np.random.random((matrix_size, matrix_size)).astype(np.float32)
        B_host = np.random.random((matrix_size, matrix_size)).astype(np.float32)
        
        # Allocate GPU memory for matrices
        with GPUTimer("Matrix GPU allocation"):
            A_gpu = cuda.to_device(A_host)
            B_gpu = cuda.to_device(B_host)
            C_gpu = cuda.device_array((matrix_size, matrix_size), dtype=np.float32)
        
        # Configure 2D kernel launch
        threads_per_block_2d = (16, 16)
        blocks_per_grid_x = (matrix_size + threads_per_block_2d[0] - 1) // threads_per_block_2d[0]
        blocks_per_grid_y = (matrix_size + threads_per_block_2d[1] - 1) // threads_per_block_2d[1]
        blocks_per_grid_2d = (blocks_per_grid_x, blocks_per_grid_y)
        
        print(f"2D Kernel config: {blocks_per_grid_2d} blocks, {threads_per_block_2d} threads per block")
        
        # Launch matrix multiplication kernel
        with GPUTimer("Custom matrix multiply kernel"):
            matrix_multiply_kernel[blocks_per_grid_2d, threads_per_block_2d](A_gpu, B_gpu, C_gpu)
        
        # Compare with NumPy
        with GPUTimer("NumPy matrix multiplication (CPU)"):
            expected_matrix = np.dot(A_host, B_host)
        
        # Get result from GPU
        result_matrix = C_gpu.copy_to_host()
        
        # Verify
        max_matrix_error = np.max(np.abs(result_matrix - expected_matrix))
        print(f"Matrix multiplication max error: {max_matrix_error:.2e}")
        
        if max_matrix_error < 1e-4:
            print("✓ Custom matrix kernel result verified")
        else:
            print("⚠ Custom matrix kernel result verification failed")
        
        return True
        
    except Exception as e:
        print(f"Error in Numba CUDA kernel example: {e}")
        traceback.print_exc()
        return False

def memory_management_optimization():
    """Demonstrate GPU memory management best practices."""
    print(f"\n{'='*60}")
    print("GPU MEMORY MANAGEMENT OPTIMIZATION")
    print(f"{'='*60}")
    
    try:
        import cupy as cp
        import numpy as np
        
        # Show initial memory state
        mempool = cp.get_default_memory_pool()
        print(f"Initial GPU memory - Used: {mempool.used_bytes() / 1024**2:.1f} MB, "
              f"Total: {mempool.total_bytes() / 1024**2:.1f} MB")
        
        # Example 1: Memory pool management
        print("\n1. Memory Pool Management:")
        
        # Create large arrays
        size = 5000000
        arrays = []
        
        for i in range(5):
            with GPUTimer(f"Allocating array {i+1}"):
                arr = cp.random.random(size, dtype=cp.float32)
                arrays.append(arr)
            
            used_mb = mempool.used_bytes() / 1024**2
            total_mb = mempool.total_bytes() / 1024**2
            print(f"  After array {i+1}: Used {used_mb:.1f} MB, Total {total_mb:.1f} MB")
        
        # Free memory explicitly
        print("\nFreeing arrays and memory pool:")
        del arrays
        
        print(f"After del: Used {mempool.used_bytes() / 1024**2:.1f} MB")
        
        # Free unused memory blocks
        mempool.free_all_blocks()
        print(f"After free_all_blocks(): Used {mempool.used_bytes() / 1024**2:.1f} MB, "
              f"Total {mempool.total_bytes() / 1024**2:.1f} MB")
        
        # Example 2: In-place operations
        print("\n2. In-place Operations vs New Array Creation:")
        
        size = 2000000
        a = cp.random.random(size, dtype=cp.float32)
        
        # Method 1: Creating new arrays (memory intensive)
        with GPUTimer("New array creation method"):
            for _ in range(10):
                b = cp.sin(a)
                c = cp.cos(b)
                d = cp.exp(c * 0.1)
                result1 = cp.sum(d)
        
        # Method 2: In-place operations (memory efficient)
        temp = cp.empty_like(a)
        with GPUTimer("In-place operations method"):
            for _ in range(10):
                cp.sin(a, out=temp)
                cp.cos(temp, out=temp)
                temp *= 0.1
                cp.exp(temp, out=temp)
                result2 = cp.sum(temp)
        
        print(f"Results match: {abs(float(result1) - float(result2)) < 1e-5}")
        
        # Example 3: Memory pinning for faster transfers
        print("\n3. Memory Pinning for Faster CPU-GPU Transfers:")
        
        size = 1000000
        
        # Regular numpy array
        regular_array = np.random.random(size).astype(np.float32)
        
        # Pinned memory array
        pinned_array = cp.cuda.alloc_pinned_memory(size * 4)  # 4 bytes per float32
        pinned_view = np.frombuffer(pinned_array, dtype=np.float32)
        pinned_view[:] = np.random.random(size).astype(np.float32)
        
        # Compare transfer speeds
        with GPUTimer("Regular array CPU->GPU transfer"):
            gpu_regular = cp.asarray(regular_array)
        
        with GPUTimer("Pinned array CPU->GPU transfer"):
            gpu_pinned = cp.asarray(pinned_view)
        
        # Verify results are the same
        print(f"Arrays are equal: {cp.allclose(gpu_regular, gpu_pinned)}")
        
        # Clean up pinned memory
        pinned_array.free()
        
        return True
        
    except Exception as e:
        print(f"Error in memory management optimization: {e}")
        traceback.print_exc()
        return False

def batch_processing_optimization():
    """Demonstrate batch processing optimization techniques."""
    print(f"\n{'='*60}")
    print("BATCH PROCESSING OPTIMIZATION")
    print(f"{'='*60}")
    
    try:
        import cupy as cp
        import numpy as np
        
        # Simulate a data processing pipeline
        def process_single_item(data):
            """Process a single data item (inefficient for GPU)."""
            result = cp.sin(data)
            result = cp.cos(result)
            result = cp.exp(result * 0.1)
            return cp.sum(result)
        
        def process_batch(batch_data):
            """Process a batch of data items (efficient for GPU)."""
            # Process entire batch at once
            result = cp.sin(batch_data)
            result = cp.cos(result)
            result = cp.exp(result * 0.1)
            return cp.sum(result, axis=1)  # Sum along each row
        
        # Generate test data
        num_items = 1000
        item_size = 10000
        
        # Create individual items
        items = [cp.random.random(item_size, dtype=cp.float32) for _ in range(num_items)]
        
        # Create batch (all items stacked)
        batch = cp.stack(items)
        
        print(f"Processing {num_items} items of size {item_size:,} each")
        
        # Method 1: Process items one by one
        with GPUTimer("Individual item processing"):
            results_individual = []
            for item in items:
                result = process_single_item(item)
                results_individual.append(float(result))
        
        # Method 2: Process as a batch
        with GPUTimer("Batch processing"):
            results_batch = process_batch(batch)
            results_batch_list = [float(x) for x in results_batch]
        
        # Verify results are the same
        max_diff = max(abs(a - b) for a, b in zip(results_individual, results_batch_list))
        print(f"Max difference between methods: {max_diff:.2e}")
        
        if max_diff < 1e-5:
            print("✓ Batch processing results verified")
        else:
            print("⚠ Batch processing results differ")
        
        # Demonstrate optimal batch sizes
        print(f"\n4. Finding Optimal Batch Sizes:")
        
        batch_sizes = [1, 10, 50, 100, 500, 1000]
        item_size = 50000
        
        print(f"{'Batch Size':>10} {'Time (ms)':>12} {'Items/sec':>12} {'Efficiency':>12}")
        print("-" * 50)
        
        baseline_time = None
        
        for batch_size in batch_sizes:
            # Create batch data
            test_batch = cp.random.random((batch_size, item_size), dtype=cp.float32)
            
            # Time the processing
            start_time = time.perf_counter()
            
            # Process multiple times for better timing
            for _ in range(10):
                results = process_batch(test_batch)
                cp.cuda.Stream.null.synchronize()  # Ensure completion
            
            end_time = time.perf_counter()
            
            total_time = (end_time - start_time) * 1000  # Convert to ms
            time_per_batch = total_time / 10
            items_per_second = (batch_size * 10) / (total_time / 1000)
            
            if baseline_time is None:
                baseline_time = time_per_batch
                efficiency = 1.0
            else:
                efficiency = baseline_time / time_per_batch * batch_sizes[0] / batch_size
            
            print(f"{batch_size:>10} {time_per_batch:>9.2f} {items_per_second:>9.0f} {efficiency:>9.2f}x")
        
        return True
        
    except Exception as e:
        print(f"Error in batch processing optimization: {e}")
        traceback.print_exc()
        return False

def data_pipeline_optimization():
    """Demonstrate data pipeline optimization with streams and async operations."""
    print(f"\n{'='*60}")
    print("DATA PIPELINE OPTIMIZATION")
    print(f"{'='*60}")
    
    try:
        import cupy as cp
        import numpy as np
        
        # Simulate a data processing pipeline with multiple stages
        def pipeline_stage_1(data):
            """First processing stage."""
            return cp.sin(data) + cp.cos(data)
        
        def pipeline_stage_2(data):
            """Second processing stage."""
            return cp.exp(data * 0.1) - cp.log(cp.abs(data) + 1)
        
        def pipeline_stage_3(data):
            """Third processing stage."""
            return cp.sum(data, axis=1, keepdims=True)
        
        # Generate test data
        num_batches = 20
        batch_size = 100000
        
        print(f"Processing {num_batches} batches of {batch_size:,} elements each")
        
        # Method 1: Sequential processing (synchronous)
        print("\n1. Sequential Processing:")
        
        with GPUTimer("Sequential pipeline"):
            total_results_seq = []
            
            for i in range(num_batches):
                # Generate data
                data = cp.random.random((10, batch_size), dtype=cp.float32)
                
                # Process through pipeline
                stage1_result = pipeline_stage_1(data)
                stage2_result = pipeline_stage_2(stage1_result)
                stage3_result = pipeline_stage_3(stage2_result)
                
                total_results_seq.append(float(cp.sum(stage3_result)))
        
        # Method 2: Using CUDA streams for overlapping operations
        print("\n2. Optimized Pipeline with Streams:")
        
        try:
            # Create CUDA streams
            stream1 = cp.cuda.Stream()
            stream2 = cp.cuda.Stream()
            
            with GPUTimer("Streamed pipeline"):
                total_results_stream = []
                
                # Pre-allocate memory for better performance
                temp_data = cp.empty((10, batch_size), dtype=cp.float32)
                stage1_buffer = cp.empty((10, batch_size), dtype=cp.float32)
                stage2_buffer = cp.empty((10, batch_size), dtype=cp.float32)
                stage3_buffer = cp.empty((10, 1), dtype=cp.float32)
                
                for i in range(num_batches):
                    with stream1:
                        # Generate data
                        cp.random.random((10, batch_size), dtype=cp.float32, out=temp_data)
                        
                        # Stage 1
                        cp.sin(temp_data, out=stage1_buffer)
                        stage1_buffer += cp.cos(temp_data)
                    
                    with stream2:
                        # Wait for stage 1 to complete
                        stream1.synchronize()
                        
                        # Stage 2
                        cp.exp(stage1_buffer * 0.1, out=stage2_buffer)
                        stage2_buffer -= cp.log(cp.abs(stage1_buffer) + 1)
                        
                        # Stage 3
                        cp.sum(stage2_buffer, axis=1, keepdims=True, out=stage3_buffer)
                        
                        total_results_stream.append(float(cp.sum(stage3_buffer)))
                
                # Synchronize all streams
                stream1.synchronize()
                stream2.synchronize()
            
            # Verify results are similar
            if len(total_results_seq) == len(total_results_stream):
                max_diff = max(abs(a - b) for a, b in zip(total_results_seq, total_results_stream))
                print(f"Max difference between methods: {max_diff:.2e}")
                
                if max_diff < 1e-3:
                    print("✓ Streamed pipeline results verified")
                else:
                    print("⚠ Streamed pipeline results differ significantly")
            
        except Exception as stream_error:
            print(f"Stream optimization failed: {stream_error}")
            print("This is normal on some systems - streams require specific GPU capabilities")
        
        # Method 3: Memory-efficient processing with generators
        print("\n3. Memory-Efficient Processing:")
        
        def data_generator(num_batches, batch_size):
            """Generator that yields data batches."""
            for i in range(num_batches):
                yield cp.random.random((10, batch_size), dtype=cp.float32)
        
        with GPUTimer("Memory-efficient pipeline"):
            total_results_efficient = []
            
            # Pre-allocate working memory
            working_buffer = None
            
            for data_batch in data_generator(num_batches, batch_size):
                if working_buffer is None:
                    working_buffer = cp.empty_like(data_batch)
                
                # Process in-place when possible
                cp.sin(data_batch, out=working_buffer)
                working_buffer += cp.cos(data_batch)
                
                # Stage 2 (reuse buffer)
                working_buffer *= 0.1
                cp.exp(working_buffer, out=working_buffer)
                temp = cp.log(cp.abs(data_batch) + 1)
                working_buffer -= temp
                
                # Stage 3
                result = cp.sum(working_buffer)
                total_results_efficient.append(float(result))
        
        print(f"Processed {len(total_results_efficient)} batches efficiently")
        
        return True
        
    except Exception as e:
        print(f"Error in data pipeline optimization: {e}")
        traceback.print_exc()
        return False

def profiling_and_debugging_tips():
    """Provide tips for profiling and debugging GPU code in IDLE."""
    print(f"\n{'='*60}")
    print("PROFILING AND DEBUGGING TIPS FOR IDLE")
    print(f"{'='*60}")
    
    print("Since IDLE lacks integrated GPU profiling, here are manual techniques:")
    
    print("\n1. TIMING TECHNIQUES:")
    print("   - Use the GPUTimer class from this script")
    print("   - Always synchronize GPU operations before timing")
    print("   - Time both computation and memory transfers separately")
    
    print("\n2. MEMORY MONITORING:")
    print("   - Use cp.get_default_memory_pool() to monitor GPU memory")
    print("   - Check memory usage before and after operations")
    print("   - Use free_all_blocks() to clean up memory")
    
    print("\n3. EXTERNAL TOOLS TO USE WITH IDLE:")
    print("   - nvidia-smi: Monitor GPU utilization in terminal")
    print("   - nvtop: Real-time GPU monitoring (Linux/Mac)")
    print("   - NVIDIA Nsight Systems: Detailed profiling (advanced)")
    
    print("\n4. DEBUGGING STRATEGIES:")
    print("   - Start with small data sizes")
    print("   - Compare GPU results with CPU equivalents")
    print("   - Use try-except blocks around GPU operations")
    print("   - Check CUDA error messages carefully")
    
    print("\n5. PERFORMANCE OPTIMIZATION CHECKLIST:")
    print("   ✓ Use appropriate data types (float32 vs float64)")
    print("   ✓ Minimize CPU-GPU memory transfers")
    print("   ✓ Use batch processing for small operations")
    print("   ✓ Prefer in-place operations when possible")
    print("   ✓ Consider memory access patterns")
    print("   ✓ Profile different batch sizes")
    
    # Demonstrate a simple profiling function
    print("\n6. SIMPLE PROFILING FUNCTION:")
    
    def profile_gpu_operation(operation_func, data, name="Operation", iterations=10):
        """Simple profiling function for GPU operations."""
        import cupy as cp
        
        times = []
        
        for i in range(iterations):
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            
            result = operation_func(data)
            
            cp.cuda.Stream.null.synchronize()
            end = time.perf_counter()
            
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"{name} - Avg: {avg_time:.4f}s, Min: {min_time:.4f}s, Max: {max_time:.4f}s")
        return result, avg_time
    
    # Example usage
    try:
        import cupy as cp
        
        test_data = cp.random.random((1000, 1000), dtype=cp.float32)
        
        def test_operation(data):
            return cp.sum(cp.sin(data) * cp.cos(data))
        
        result, avg_time = profile_gpu_operation(test_operation, test_data, "Sin*Cos Sum")
        print(f"Result: {float(result):.6f}")
        
    except Exception as e:
        print(f"Profiling example failed: {e}")
    
    return True

def main():
    """Main function to run all optimization examples."""
    print("Advanced GPU Optimization Examples")
    print("=" * 50)
    
    # Check dependencies
    print("\nChecking dependencies...")
    libs, all_available = check_dependencies()
    
    if not all_available:
        print("\nCannot proceed without required libraries.")
        return
    
    # Run optimization examples
    results = {}
    
    print(f"\n{'='*60}")
    print("RUNNING OPTIMIZATION EXAMPLES")
    print(f"{'='*60}")
    
    # Custom CUDA kernels
    results["Numba CUDA Kernels"] = numba_cuda_kernel_example()
    
    # Memory management
    results["Memory Management"] = memory_management_optimization()
    
    # Batch processing
    results["Batch Processing"] = batch_processing_optimization()
    
    # Data pipeline optimization
    results["Data Pipeline"] = data_pipeline_optimization()
    
    # Profiling tips
    results["Profiling Tips"] = profiling_and_debugging_tips()
    
    # Summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION EXAMPLES SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"Examples completed: {passed_tests}/{total_tests}")
    
    for example_name, passed in results.items():
        status = "✓ COMPLETED" if passed else "✗ FAILED"
        print(f"{example_name}: {status}")
    
    print(f"\n{'='*60}")
    print("KEY TAKEAWAYS:")
    print("1. Custom CUDA kernels provide maximum control and performance")
    print("2. Proper memory management is crucial for GPU efficiency")
    print("3. Batch processing dramatically improves GPU utilization")
    print("4. Pipeline optimization can overlap computation and data transfer")
    print("5. Use external tools for detailed profiling beyond IDLE's capabilities")
    
    print(f"\nFor more advanced GPU programming, consider:")
    print("- NVIDIA Nsight Compute for kernel profiling")
    print("- CuPy's profiling utilities")
    print("- Moving to a more advanced IDE like PyCharm or VS Code")

if __name__ == "__main__":
    main()
