"""
Benchmark scaled_dot_product_attention at different scales.

This script tests the attention mechanism with:
- Fixed batch size of 8
- No multihead attention (single head)
- Varying head embedding dimensions: [16, 32, 64, 128]
- Varying sequence lengths: [256, 1024, 4096, 8192, 16384]
"""

import torch
import torch.nn.functional as F
from cs336_basics.model import scaled_dot_product_attention
import time
import gc
import itertools
import pandas as pd
from typing import Dict, List, Any

# Configuration
BATCH_SIZE = 8
D_MODEL_VALUES = [16, 32, 64, 128]
SEQ_LENGTH_VALUES = [256, 1024, 4096, 8192, 16384]
NUM_FORWARD_PASSES = 100
NUM_BACKWARD_PASSES = 100
WARMUP_ITERATIONS = 10

def cuda_synchronize():
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def get_memory_allocated():
    """Get current GPU memory allocation in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
    return 0

def get_memory_reserved():
    """Get current GPU memory reservation in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 * 1024)  # Convert to MB
    return 0

def benchmark_attention(d_model: int, seq_len: int, device: str = 'cuda') -> Dict[str, Any]:
    """
    Benchmark attention at a specific configuration.
    
    Args:
        d_model: Head embedding dimension
        seq_len: Sequence length
        device: Device to run on
        
    Returns:
        Dictionary with benchmark results
    """
    result = {
        'd_model': d_model,
        'seq_len': seq_len,
        'batch_size': BATCH_SIZE,
        'status': 'success',
        'forward_time': None,
        'backward_time': None,
        'memory_before_backward_mb': None,
        'peak_memory_mb': None,
        'error': None
    }
    
    try:
        # Clear GPU cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        # Create random inputs
        # Shape: (batch_size, seq_len, d_model)
        Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
        K = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
        V = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
        
        # Create causal mask (optional - for causal attention)
        # Shape: (batch_size, seq_len, seq_len)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).expand(BATCH_SIZE, -1, -1)
        mask = ~mask  # Invert: True where attention is allowed
        
        print(f"  Created tensors - Q/K/V shape: {Q.shape}, Mask shape: {mask.shape}")
        
        # Warmup
        print(f"  Running warmup ({WARMUP_ITERATIONS} iterations)...")
        for _ in range(WARMUP_ITERATIONS):
            output = scaled_dot_product_attention(Q, K, V, mask)
            cuda_synchronize()
            if _ == 0:  # Prepare for backward on first iteration
                loss = output.sum()
                loss.backward()
                Q.grad = None
                K.grad = None
                V.grad = None
        
        # Forward pass benchmark
        print(f"  Benchmarking forward pass ({NUM_FORWARD_PASSES} iterations)...")
        cuda_synchronize()
        start_time = time.time()
        
        for _ in range(NUM_FORWARD_PASSES):
            output = scaled_dot_product_attention(Q, K, V, mask)
            cuda_synchronize()
        
        forward_time = (time.time() - start_time) / NUM_FORWARD_PASSES
        result['forward_time'] = forward_time * 1000  # Convert to ms
        
        # Prepare for backward pass
        output = scaled_dot_product_attention(Q, K, V, mask)
        loss = output.sum()
        
        # Measure memory before backward
        cuda_synchronize()
        result['memory_before_backward_mb'] = get_memory_allocated()
        
        # Backward pass benchmark
        print(f"  Benchmarking backward pass ({NUM_BACKWARD_PASSES} iterations)...")
        cuda_synchronize()
        start_time = time.time()
        
        for _ in range(NUM_BACKWARD_PASSES):
            # Clear gradients
            Q.grad = None
            K.grad = None
            V.grad = None
            
            # Recompute forward and backward
            output = scaled_dot_product_attention(Q, K, V, mask)
            loss = output.sum()
            loss.backward()
            cuda_synchronize()
        
        backward_time = (time.time() - start_time) / NUM_BACKWARD_PASSES
        result['backward_time'] = backward_time * 1000  # Convert to ms
        
        # Record peak memory
        if torch.cuda.is_available():
            result['peak_memory_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        print(f"  ✓ Forward: {result['forward_time']:.2f}ms, Backward: {result['backward_time']:.2f}ms")
        
    except torch.cuda.OutOfMemoryError as e:
        result['status'] = 'OOM'
        result['error'] = str(e)
        print(f"  ✗ Out of Memory!")
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        print(f"  ✗ Error: {e}")
    
    finally:
        # Clean up
        if 'Q' in locals():
            del Q
        if 'K' in locals():
            del K
        if 'V' in locals():
            del V
        if 'output' in locals():
            del output
        if 'loss' in locals():
            del loss
        if 'mask' in locals():
            del mask
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return result

def main():
    """Main benchmark function."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("Attention Mechanism Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_memory:.2f} GB")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"D_model values: {D_MODEL_VALUES}")
    print(f"Sequence lengths: {SEQ_LENGTH_VALUES}")
    print(f"Forward passes per config: {NUM_FORWARD_PASSES}")
    print(f"Backward passes per config: {NUM_BACKWARD_PASSES}")
    print("=" * 80)
    print()
    
    results = []
    
    # Iterate through all configurations
    for d_model, seq_len in itertools.product(D_MODEL_VALUES, SEQ_LENGTH_VALUES):
        print(f"Testing d_model={d_model}, seq_len={seq_len}:")
        
        result = benchmark_attention(d_model, seq_len, device)
        results.append(result)
        
        # Add delay between tests to allow GPU to cool down
        time.sleep(0.5)
        print()
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Display results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    
    # Format the results for better display
    display_columns = ['d_model', 'seq_len', 'status', 'forward_time', 'backward_time', 
                      'memory_before_backward_mb', 'peak_memory_mb']
    
    # Create a pivot table for forward times
    print("\nForward Pass Times (ms):")
    print("-" * 40)
    forward_pivot = df[df['status'] == 'success'].pivot_table(
        values='forward_time', 
        index='seq_len', 
        columns='d_model'
    )
    print(forward_pivot.to_string(float_format='%.2f', na_rep='OOM'))
    
    # Create a pivot table for backward times
    print("\nBackward Pass Times (ms):")
    print("-" * 40)
    backward_pivot = df[df['status'] == 'success'].pivot_table(
        values='backward_time', 
        index='seq_len', 
        columns='d_model'
    )
    print(backward_pivot.to_string(float_format='%.2f', na_rep='OOM'))
    
    # Create a pivot table for memory usage
    print("\nPeak Memory Usage (MB):")
    print("-" * 40)
    memory_pivot = df[df['status'] == 'success'].pivot_table(
        values='peak_memory_mb', 
        index='seq_len', 
        columns='d_model'
    )
    print(memory_pivot.to_string(float_format='%.1f', na_rep='OOM'))
    
    # Show OOM configurations
    oom_configs = df[df['status'] == 'OOM']
    if not oom_configs.empty:
        print("\nOut-of-Memory Configurations:")
        print("-" * 40)
        for _, row in oom_configs.iterrows():
            print(f"  d_model={row['d_model']}, seq_len={row['seq_len']}")
    
    # Save detailed results to CSV
    output_file = 'attention_benchmark_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    # Print summary statistics
    successful = df[df['status'] == 'success']
    if not successful.empty:
        print("\nSummary Statistics (successful runs):")
        print("-" * 40)
        print(f"Fastest forward pass: {successful['forward_time'].min():.2f}ms")
        print(f"Slowest forward pass: {successful['forward_time'].max():.2f}ms")
        print(f"Fastest backward pass: {successful['backward_time'].min():.2f}ms")
        print(f"Slowest backward pass: {successful['backward_time'].max():.2f}ms")
        print(f"Min peak memory: {successful['peak_memory_mb'].min():.1f}MB")
        print(f"Max peak memory: {successful['peak_memory_mb'].max():.1f}MB")

if __name__ == "__main__":
    main()