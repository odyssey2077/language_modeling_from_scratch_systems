from cs336_basics.model import BasicsTransformerLM
import torch
import torch.nn as nn
import timeit
import numpy as np
import pandas as pd
import gc
import argparse

VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
BATCH_SIZE = 4

def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Model configurations
model_configs = {
    'small': {'d_model': 768, 'd_ff': 3072, 'num_layers': 12, 'num_heads': 12},
    'medium': {'d_model': 1024, 'd_ff': 4096, 'num_layers': 24, 'num_heads': 16},
    'large': {'d_model': 1280, 'd_ff': 5120, 'num_layers': 36, 'num_heads': 20},
    'xl': {'d_model': 1600, 'd_ff': 6400, 'num_layers': 48, 'num_heads': 25},
    '2.7B': {'d_model': 2560, 'd_ff': 10240, 'num_layers': 32, 'num_heads': 32},
}

def create_model(d_model, d_ff, num_layers, num_heads, device='cuda'):
    return BasicsTransformerLM(
        vocab_size=VOCAB_SIZE, 
        context_length=CONTEXT_LENGTH, 
        d_model=d_model, 
        num_layers=num_layers, 
        num_heads=num_heads, 
        d_ff=d_ff, 
        rope_theta=10000
    ).to(device=device)

def benchmark_forward(model, x, warmup, n_trials=10):
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
        cuda_synchronize()
    
    # Benchmark
    times = []
    for _ in range(n_trials):
        cuda_synchronize()
        start = timeit.default_timer()
        
        with torch.no_grad():
            _ = model(x)
        
        cuda_synchronize()
        end = timeit.default_timer()
        times.append(end - start)
    
    return np.mean(times), np.std(times)

def benchmark_backward(model, x, warmup, n_trials=10):
    # Warmup
    for _ in range(warmup):
        model.zero_grad()
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        cuda_synchronize()
    
    # Benchmark
    times = []
    for _ in range(n_trials):
        model.zero_grad()
        cuda_synchronize()
        
        # Forward pass
        logits = model(x)
        loss = logits.sum()
        
        # Time only the backward pass
        cuda_synchronize()
        start = timeit.default_timer()
        
        loss.backward()
        
        cuda_synchronize()
        end = timeit.default_timer()
        times.append(end - start)
    
    return np.mean(times), np.std(times)

def main():
    parser = argparse.ArgumentParser(description='Run transformer model benchmarks')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup rounds (default: 5)')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmarks on {device}")
    print(f"Batch size: {BATCH_SIZE}, Context length: {CONTEXT_LENGTH}")
    print(f"Warmup rounds: {args.warmup}")
    print()
    
    results = []
    
    for model_name, config in model_configs.items():
        print(f"Benchmarking {model_name} model...")
        
        # Create model
        model = create_model(**config, device=device)
        model.eval()
        
        # Create input
        x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)
        
        # Benchmark forward pass
        forward_mean, forward_std = benchmark_forward(model, x, warmup=args.warmup)
        
        # Benchmark backward pass
        backward_mean, backward_std = benchmark_backward(model, x, warmup=args.warmup)
        
        # Store results
        results.append({
            'Model': model_name,
            'd_model': config['d_model'],
            'd_ff': config['d_ff'],
            'num_layers': config['num_layers'],
            'num_heads': config['num_heads'],
            'vocab_size': VOCAB_SIZE,
            'context_length': CONTEXT_LENGTH,
            'batch_size': BATCH_SIZE,
            'Forward Mean (s)': forward_mean,
            'Forward Std (s)': forward_std,
            'Backward Mean (s)': backward_mean,
            'Backward Std (s)': backward_std,
            'Total Mean (s)': forward_mean + backward_mean,
            'Total Std (s)': np.sqrt(forward_std**2 + backward_std**2)
        })
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"  Forward:  {forward_mean:.4f} ± {forward_std:.4f} seconds")
        print(f"  Backward: {backward_mean:.4f} ± {backward_std:.4f} seconds")
        print(f"  Total:    {forward_mean + backward_mean:.4f} seconds")
        print()
    
    # Create DataFrame and display
    df = pd.DataFrame(results)
    print("\nSummary Results:")
    print("=" * 80)
    print(df.to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    main()