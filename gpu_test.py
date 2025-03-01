import tensorflow as tf
import time
import matplotlib.pyplot as plt
import numpy as np

# Check TensorFlow version and GPU availability
print("TensorFlow version:", tf.__version__)
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(physical_devices))
for device in physical_devices:
    print(f"Found GPU: {device}")
    # Enable memory growth to avoid allocating all GPU memory at once
    tf.config.experimental.set_memory_growth(device, True)

# Quick performance test if GPU is available
if len(physical_devices) > 0:
    print("\nRunning performance test...")
    
    # Matrix sizes to test
    sizes = [1000, 2000, 3000, 4000, 5000]
    times = []
    
    for size in sizes:
        # Create random matrices
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])
        
        # Time matrix multiplication
        start = time.time()
        c = tf.matmul(a, b)
        # Force execution
        result = c.numpy()
        end = time.time()
        
        elapsed = end - start
        times.append(elapsed)
        print(f"Matrix size {size}x{size}: {elapsed:.4f} seconds")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times, 'o-', linewidth=2)
    plt.title('Matrix Multiplication Performance on GPU')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.savefig('gpu_performance.png')
    print("Performance plot saved as 'gpu_performance.png'")
else:
    print("No GPU detected for testing!")
    
    # Check system environment variables
    print("\nSystem Environment Variables:")
    import os
    cuda_path = os.environ.get('CUDA_PATH', 'Not set')
    print(f"CUDA_PATH: {cuda_path}")
    
    # Check if CUDA executables can be found
    print("\nTrying to locate CUDA executables:")
    import subprocess
    try:
        nvcc_output = subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT).decode()
        print("nvcc found:")
        print(nvcc_output)
    except:
        print("nvcc not found in PATH")
    
    try:
        smi_output = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode()
        print("nvidia-smi found:")
        print(smi_output)
    except:
        print("nvidia-smi not found in PATH")