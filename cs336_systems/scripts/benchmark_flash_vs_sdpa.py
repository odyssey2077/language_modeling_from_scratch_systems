#!/usr/bin/env python3
import torch
import time
import numpy as np
import sys
import os

sys.path.append('/Users/wheatwaves/dev/assignment2-systems')
sys.path.append('/Users/wheatwaves/dev/assignment2-systems/cs336-basics')

from cs336_systems.flash_attention import FlashAttentionV2
from cs336_basics.model import scaled_dot_product_attention

def benchmark_attention(batch_size, seq_len, d_model, num_warmup=5, num_trials=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    K = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    V = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float16)
    
    
    # Benchmark FlashAttentionV2
    flash_times = []
    for _ in range(num_warmup):
        try:
            _ = FlashAttentionV2.apply(Q, K, V)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        except Exception as e:
            print(f"FlashAttentionV2 failed for batch={batch_size}, seq={seq_len}, d={d_model}: {e}")
            return None, None
    
    for _ in range(num_trials):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        try:
            _ = FlashAttentionV2.apply(Q, K, V)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            flash_times.append(end_time - start_time)
        except Exception as e:
            print(f"FlashAttentionV2 failed during timing for batch={batch_size}, seq={seq_len}, d={d_model}: {e}")
            return None, None
    
    # Benchmark scaled_dot_product_attention
    sdpa_times = []
    for _ in range(num_warmup):
        _ = scaled_dot_product_attention(Q, K, V)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    for _ in range(num_trials):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = scaled_dot_product_attention(Q, K, V)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        sdpa_times.append(end_time - start_time)
    
    return np.mean(flash_times), np.mean(sdpa_times)

def main():
    print("Benchmarking FlashAttentionV2 vs scaled_dot_product_attention")
    print("=" * 80)
    
    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_models = [16, 32, 64, 128, 256]
    batch_sizes = [1]
    
    results = []
    
    print(f"{'Batch':<6} {'SeqLen':<8} {'DModel':<8} {'FlashV2(ms)':<12} {'SDPA(ms)':<12} {'Speedup':<10}")
    print("-" * 80)
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            for d_model in d_models:
                # Skip if sequence length is not divisible by tile size
                if seq_len % 128 != 0:
                    continue
                    
                flash_time, sdpa_time = benchmark_attention(batch_size, seq_len, d_model)
                
                if flash_time is not None and sdpa_time is not None:
                    speedup = sdpa_time / flash_time if flash_time > 0 else float('inf')
                    
                    print(f"{batch_size:<6} {seq_len:<8} {d_model:<8} {flash_time*1000:<12.3f} {sdpa_time*1000:<12.3f} {speedup:<10.2f}")
                    
                    results.append({
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'd_model': d_model,
                        'flash_time_ms': flash_time * 1000,
                        'sdpa_time_ms': sdpa_time * 1000,
                        'speedup': speedup
                    })
                else:
                    print(f"{batch_size:<6} {seq_len:<8} {d_model:<8} {'FAILED':<12} {'FAILED':<12} {'N/A':<10}")
    
    # Summary statistics
    if results:
        speedups = [r['speedup'] for r in results if r['speedup'] != float('inf')]
        print("\n" + "=" * 80)
        print("SUMMARY:")
        print(f"Total configurations tested: {len(results)}")
        print(f"Average speedup: {np.mean(speedups):.2f}x")
        print(f"Median speedup: {np.median(speedups):.2f}x")
        print(f"Max speedup: {max(speedups):.2f}x")
        print(f"Min speedup: {min(speedups):.2f}x")

if __name__ == "__main__":
    main()