#!/usr/bin/env python3
"""
GPU Monitoring Utilities for IDLE
=================================

This module provides utility functions for monitoring GPU performance
and memory usage that can be easily imported into other scripts.

Author: Python GPU Guide
"""

import time
import sys
from typing import Dict, List, Optional, Any, Tuple
import contextlib

def get_gpu_info(method='auto'):
    """
    Get GPU information using the best available method.
    
    Args:
        method (str): 'auto', 'pynvml', or 'gputil'
    
    Returns:
        tuple: (gpu_info_list, method_used)
    """
    if method == 'auto':
        # Try pynvml first, then GPUtil
        gpus, used_method = get_gpu_info_pynvml()
        if not gpus:
            gpus, used_method = get_gpu_info_gputil()
        return gpus, used_method
    elif method == 'pynvml':
        return get_gpu_info_pynvml()
    elif method == 'gputil':
        return get_gpu_info_gputil()
    else:
        raise ValueError("Method must be 'auto', 'pynvml', or 'gputil'")

def get_gpu_info_pynvml():
    """Get GPU info using nvidia-ml-py3."""
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Basic info
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = mem_info.total // 1024**2
            memory_used = mem_info.used // 1024**2
            memory_free = mem_info.free // 1024**2
            
            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                memory_util = util.memory
            except:
                gpu_util = None
                memory_util = None
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = None
            
            # Power usage
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000
            except:
                power = None
            
            gpu_info = {
                'index': i,
                'name': name,
                'memory_total_mb': memory_total,
                'memory_used_mb': memory_used,
                'memory_free_mb': memory_free,
                'memory_percent': round((memory_used / memory_total) * 100, 1),
                'gpu_utilization_percent': gpu_util,
                'memory_utilization_percent': memory_util,
                'temperature_c': temp,
                'power_watts': power,
            }
            gpus.append(gpu_info)
        
        pynvml.nvmlShutdown()
        return gpus, "pynvml"
        
    except Exception as e:
        return [], f"pynvml error: {str(e)}"

def get_gpu_info_gputil():
    """Get GPU info using GPUtil."""
    try:
        import GPUtil
        gpus_raw = GPUtil.getGPUs()
        gpus = []
        
        for gpu in gpus_raw:
            gpu_info = {
                'index': gpu.id,
                'name': gpu.name,
                'memory_total_mb': round(gpu.memoryTotal),
                'memory_used_mb': round(gpu.memoryUsed),
                'memory_free_mb': round(gpu.memoryFree),
                'memory_percent': round(gpu.memoryUtil * 100, 1),
                'gpu_utilization_percent': round(gpu.load * 100, 1),
                'temperature_c': gpu.temperature,
            }
            gpus.append(gpu_info)
        
        return gpus, "GPUtil"
        
    except Exception as e:
        return [], f"GPUtil error: {str(e)}"

def print_gpu_status(gpus: List[Dict], compact: bool = False):
    """
    Print GPU status in a formatted way.
    
    Args:
        gpus: List of GPU info dictionaries
        compact: If True, print in compact format
    """
    if not gpus:
        print("No GPU information available")
        return
    
    if compact:
        for gpu in gpus:
            util = gpu.get('gpu_utilization_percent', 'N/A')
            mem_pct = gpu.get('memory_percent', 'N/A')
            temp = gpu.get('temperature_c', 'N/A')
            print(f"GPU{gpu['index']}: {util}% | Mem: {mem_pct}% | Temp: {temp}°C")
    else:
        for gpu in gpus:
            print(f"\nGPU {gpu['index']}: {gpu['name']}")
            print(f"  Memory: {gpu['memory_used_mb']:,}MB / {gpu['memory_total_mb']:,}MB ({gpu['memory_percent']}%)")
            
            if gpu.get('gpu_utilization_percent') is not None:
                print(f"  GPU Utilization: {gpu['gpu_utilization_percent']}%")
            
            if gpu.get('temperature_c') is not None:
                print(f"  Temperature: {gpu['temperature_c']}°C")
            
            if gpu.get('power_watts') is not None:
                print(f"  Power: {gpu['power_watts']}W")

class GPUMonitor:
    """Context manager for monitoring GPU usage during code execution."""
    
    def __init__(self, interval: float = 1.0, print_stats: bool = True):
        self.interval = interval
        self.print_stats = print_stats
        self.initial_stats = None
        self.final_stats = None
        self.monitoring = False
    
    def __enter__(self):
        self.initial_stats, _ = get_gpu_info()
        if self.print_stats and self.initial_stats:
            print("Initial GPU state:")
            print_gpu_status(self.initial_stats, compact=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_stats, _ = get_gpu_info()
        if self.print_stats and self.final_stats:
            print("Final GPU state:")
            print_gpu_status(self.final_stats, compact=True)
            
            # Show memory usage change
            if self.initial_stats and len(self.initial_stats) == len(self.final_stats):
                for i, (initial, final) in enumerate(zip(self.initial_stats, self.final_stats)):
                    mem_change = final['memory_used_mb'] - initial['memory_used_mb']
                    if abs(mem_change) > 1:  # Only show if change > 1MB
                        print(f"GPU{i} memory change: {mem_change:+.0f}MB")

def get_cupy_memory_info():
    """Get CuPy memory pool information if available."""
    try:
        import cupy as cp
        mempool = cp.get_default_memory_pool()
        
        info = {
            'used_bytes': mempool.used_bytes(),
            'total_bytes': mempool.total_bytes(),
            'used_mb': mempool.used_bytes() / 1024**2,
            'total_mb': mempool.total_bytes() / 1024**2,
        }
        return info
    except ImportError:
        return None
    except Exception as e:
        return {'error': str(e)}

def print_cupy_memory_status():
    """Print CuPy memory pool status."""
    info = get_cupy_memory_info()
    if info is None:
        print("CuPy not available")
    elif 'error' in info:
        print(f"CuPy memory error: {info['error']}")
    else:
        print(f"CuPy Memory Pool: {info['used_mb']:.1f}MB / {info['total_mb']:.1f}MB used")

def monitor_gpu_realtime(duration: float = 10.0, interval: float = 1.0):
    """
    Monitor GPU usage in real-time.
    
    Args:
        duration: How long to monitor (seconds)
        interval: Update interval (seconds)
    """
    print(f"Monitoring GPU for {duration} seconds (interval: {interval}s)")
    print("Press Ctrl+C to stop early")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            gpus, method = get_gpu_info()
            
            if gpus:
                timestamp = time.strftime('%H:%M:%S')
                print(f"\r[{timestamp}] ", end="")
                
                for gpu in gpus:
                    util = gpu.get('gpu_utilization_percent', 'N/A')
                    mem_pct = gpu.get('memory_percent', 'N/A')
                    temp = gpu.get('temperature_c', 'N/A')
                    print(f"GPU{gpu['index']}: {util}% | Mem: {mem_pct}% | {temp}°C  ", end="")
                
                print("", end="", flush=True)
            else:
                print(f"\r[{time.strftime('%H:%M:%S')}] No GPU data available", end="", flush=True)
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    print("\nMonitoring complete")

def check_gpu_availability():
    """
    Check if GPU acceleration is available and working.
    
    Returns:
        dict: Status information about GPU availability
    """
    status = {
        'nvidia_ml_available': False,
        'gputil_available': False,
        'cupy_available': False,
        'numba_cuda_available': False,
        'gpu_detected': False,
        'cuda_working': False,
    }
    
    # Check nvidia-ml-py3
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            status['nvidia_ml_available'] = True
            status['gpu_detected'] = True
        pynvml.nvmlShutdown()
    except:
        pass
    
    # Check GPUtil
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            status['gputil_available'] = True
            status['gpu_detected'] = True
    except:
        pass
    
    # Check CuPy
    try:
        import cupy as cp
        # Try a simple operation
        test_array = cp.array([1, 2, 3])
        result = cp.sum(test_array)
        status['cupy_available'] = True
        status['cuda_working'] = True
    except:
        pass
    
    # Check Numba CUDA
    try:
        from numba import cuda
        cuda.detect()
        status['numba_cuda_available'] = True
    except:
        pass
    
    return status

def print_gpu_availability_report():
    """Print a comprehensive report of GPU availability."""
    print("GPU Availability Report")
    print("=" * 30)
    
    status = check_gpu_availability()
    
    print(f"NVIDIA ML Python:  {'✓' if status['nvidia_ml_available'] else '✗'}")
    print(f"GPUtil:            {'✓' if status['gputil_available'] else '✗'}")
    print(f"CuPy:              {'✓' if status['cupy_available'] else '✗'}")
    print(f"Numba CUDA:        {'✓' if status['numba_cuda_available'] else '✗'}")
    print(f"GPU Detected:      {'✓' if status['gpu_detected'] else '✗'}")
    print(f"CUDA Working:      {'✓' if status['cuda_working'] else '✗'}")
    
    if status['cuda_working']:
        print("\n🎉 GPU acceleration is ready to use!")
    elif status['gpu_detected']:
        print("\n⚠ GPU detected but CUDA libraries may need installation")
    else:
        print("\n❌ No GPU detected or drivers not installed")

# Example usage functions
def example_basic_monitoring():
    """Example of basic GPU monitoring."""
    print("Basic GPU Monitoring Example")
    print("-" * 30)
    
    # Get current GPU status
    gpus, method = get_gpu_info()
    print(f"Using method: {method}")
    print_gpu_status(gpus)
    
    # Show CuPy memory if available
    print_cupy_memory_status()

def example_context_monitoring():
    """Example of using GPU monitoring context manager."""
    print("Context Manager Monitoring Example")
    print("-" * 35)
    
    try:
        import cupy as cp
        
        with GPUMonitor():
            print("Creating large array...")
            large_array = cp.random.random((5000, 5000), dtype=cp.float32)
            
            print("Performing computation...")
            result = cp.sum(cp.sin(large_array) * cp.cos(large_array))
            
            print(f"Result: {float(result):.6f}")
            
            # Clean up
            del large_array
    
    except ImportError:
        print("CuPy not available for this example")

if __name__ == "__main__":
    # Run examples when script is executed directly
    print_gpu_availability_report()
    print()
    example_basic_monitoring()
    print()
    example_context_monitoring()
