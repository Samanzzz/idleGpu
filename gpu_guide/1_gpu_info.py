#!/usr/bin/env python3
"""
GPU Information Retrieval Script for IDLE
==========================================

This script provides comprehensive GPU information that can be run directly in IDLE.
It uses multiple methods to gather GPU data and provides fallbacks for different systems.

Required packages (install with pip):
    pip install nvidia-ml-py3 GPUtil psutil

Author: Python GPU Guide
"""

import sys
import time
import platform
from typing import Dict, List, Optional, Any

def check_and_import_libraries():
    """Check for required libraries and provide installation instructions."""
    missing_libs = []
    available_libs = {}
    
    # Check nvidia-ml-py3 (pynvml)
    try:
        import pynvml
        available_libs['pynvml'] = pynvml
        print("✓ nvidia-ml-py3 (pynvml) available")
    except ImportError:
        missing_libs.append("nvidia-ml-py3")
        print("✗ nvidia-ml-py3 not found")
    
    # Check GPUtil
    try:
        import GPUtil
        available_libs['GPUtil'] = GPUtil
        print("✓ GPUtil available")
    except ImportError:
        missing_libs.append("GPUtil")
        print("✗ GPUtil not found")
    
    # Check psutil
    try:
        import psutil
        available_libs['psutil'] = psutil
        print("✓ psutil available")
    except ImportError:
        missing_libs.append("psutil")
        print("✗ psutil not found")
    
    if missing_libs:
        print(f"\nMissing libraries: {', '.join(missing_libs)}")
        print("Install with: pip install " + " ".join(missing_libs))
        print("Then restart IDLE and run this script again.\n")
    
    return available_libs, len(missing_libs) == 0

def get_system_info():
    """Get basic system information."""
    info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'processor': platform.processor(),
        'python_version': sys.version,
    }
    return info

def get_gpu_info_pynvml():
    """Get GPU information using nvidia-ml-py3 (most comprehensive for NVIDIA)."""
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
            memory_total = mem_info.total // 1024**2  # Convert to MB
            memory_used = mem_info.used // 1024**2
            memory_free = mem_info.free // 1024**2
            
            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                memory_util = util.memory
            except:
                gpu_util = "N/A"
                memory_util = "N/A"
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = "N/A"
            
            # Driver version
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            except:
                driver_version = "N/A"
            
            # Power usage
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert to watts
            except:
                power = "N/A"
            
            # Clock speeds
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except:
                graphics_clock = "N/A"
                memory_clock = "N/A"
            
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
                'graphics_clock_mhz': graphics_clock,
                'memory_clock_mhz': memory_clock,
                'driver_version': driver_version
            }
            gpus.append(gpu_info)
        
        pynvml.nvmlShutdown()
        return gpus, "pynvml"
        
    except Exception as e:
        return [], f"pynvml error: {str(e)}"

def get_gpu_info_gputil():
    """Get GPU information using GPUtil (simpler, cross-platform)."""
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
                'driver_version': gpu.driver
            }
            gpus.append(gpu_info)
        
        return gpus, "GPUtil"
        
    except Exception as e:
        return [], f"GPUtil error: {str(e)}"

def format_gpu_info(gpus: List[Dict], method: str):
    """Format GPU information for display."""
    if not gpus:
        return "No GPU information available."
    
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"GPU INFORMATION (via {method})")
    output.append(f"{'='*60}")
    
    for gpu in gpus:
        output.append(f"\nGPU {gpu['index']}: {gpu['name']}")
        output.append("-" * 50)
        
        # Memory information
        output.append(f"Memory Total:     {gpu['memory_total_mb']:,} MB")
        output.append(f"Memory Used:      {gpu['memory_used_mb']:,} MB ({gpu['memory_percent']}%)")
        output.append(f"Memory Free:      {gpu['memory_free_mb']:,} MB")
        
        # Utilization
        if 'gpu_utilization_percent' in gpu and gpu['gpu_utilization_percent'] != "N/A":
            output.append(f"GPU Utilization:  {gpu['gpu_utilization_percent']}%")
        
        if 'memory_utilization_percent' in gpu and gpu['memory_utilization_percent'] != "N/A":
            output.append(f"Mem Utilization:  {gpu['memory_utilization_percent']}%")
        
        # Temperature
        if 'temperature_c' in gpu and gpu['temperature_c'] != "N/A":
            output.append(f"Temperature:      {gpu['temperature_c']}°C")
        
        # Power
        if 'power_watts' in gpu and gpu['power_watts'] != "N/A":
            output.append(f"Power Usage:      {gpu['power_watts']} W")
        
        # Clock speeds
        if 'graphics_clock_mhz' in gpu and gpu['graphics_clock_mhz'] != "N/A":
            output.append(f"Graphics Clock:   {gpu['graphics_clock_mhz']} MHz")
        
        if 'memory_clock_mhz' in gpu and gpu['memory_clock_mhz'] != "N/A":
            output.append(f"Memory Clock:     {gpu['memory_clock_mhz']} MHz")
        
        # Driver version
        if 'driver_version' in gpu and gpu['driver_version'] != "N/A":
            output.append(f"Driver Version:   {gpu['driver_version']}")
    
    return "\n".join(output)

def monitor_gpu_realtime(duration_seconds=10, interval_seconds=1):
    """Monitor GPU usage in real-time for a specified duration."""
    print(f"\nReal-time GPU monitoring for {duration_seconds} seconds...")
    print("(This is useful for monitoring GPU usage while running other scripts)")
    print("-" * 60)
    
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        # Try pynvml first, then GPUtil
        gpus, method = get_gpu_info_pynvml()
        if not gpus:
            gpus, method = get_gpu_info_gputil()
        
        if gpus:
            print(f"\r[{time.strftime('%H:%M:%S')}] ", end="")
            for i, gpu in enumerate(gpus):
                util = gpu.get('gpu_utilization_percent', 'N/A')
                mem_percent = gpu.get('memory_percent', 'N/A')
                temp = gpu.get('temperature_c', 'N/A')
                
                print(f"GPU{i}: {util}% | Mem: {mem_percent}% | Temp: {temp}°C  ", end="")
            print("", end="", flush=True)
        else:
            print(f"\r[{time.strftime('%H:%M:%S')}] No GPU data available", end="", flush=True)
        
        time.sleep(interval_seconds)
    
    print("\nMonitoring complete.")

def main():
    """Main function to demonstrate GPU information retrieval."""
    print("Python GPU Information Script")
    print("=" * 40)
    
    # Check system info
    print("\nSYSTEM INFORMATION:")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Check available libraries
    print("\nLIBRARY STATUS:")
    available_libs, all_available = check_and_import_libraries()
    
    if not all_available:
        print("\nSome libraries are missing. Install them and try again.")
        return
    
    # Try to get GPU information using different methods
    print("\nATTEMPTING GPU DETECTION...")
    
    # Method 1: pynvml (most comprehensive for NVIDIA)
    gpus, method = get_gpu_info_pynvml()
    if gpus:
        print(format_gpu_info(gpus, method))
    else:
        print(f"pynvml failed: {method}")
        
        # Method 2: GPUtil (fallback)
        gpus, method = get_gpu_info_gputil()
        if gpus:
            print(format_gpu_info(gpus, method))
        else:
            print(f"GPUtil failed: {method}")
            print("\nNo GPU detected or drivers not properly installed.")
            print("Make sure you have:")
            print("1. NVIDIA GPU with proper drivers installed")
            print("2. CUDA toolkit installed (optional but recommended)")
            print("3. Run 'nvidia-smi' in terminal to verify GPU is detected")
            return
    
    # Offer real-time monitoring
    print("\n" + "="*60)
    response = input("Would you like to monitor GPU usage in real-time for 10 seconds? (y/n): ")
    if response.lower().startswith('y'):
        monitor_gpu_realtime()
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("1. Run '2_cpu_vs_gpu_demo.py' to see CPU vs GPU performance comparison")
    print("2. Check 'nvidia-smi' command in terminal for additional GPU info")
    print("3. Use this script as a template for your own GPU monitoring needs")

if __name__ == "__main__":
    main()
