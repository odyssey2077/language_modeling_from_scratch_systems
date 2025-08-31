from cs336_basics.model import BasicsTransformerLM, scaled_dot_product_attention
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx
import timeit
import numpy as np
import pandas as pd
import gc
import argparse
import math
from torch import einsum
from torch.nn.functional import softmax
from jaxtyping import Float
from torch import Tensor
from typing import Optional

VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
BATCH_SIZE = 4

def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Annotated version of scaled_dot_product_attention with NVTX ranges
@nvtx.range("scaled_dot_product_attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Optional[Float[Tensor, " ... queries keys"]] = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]
    
    with nvtx.range("computing_attention_scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    
    with nvtx.range("computing_softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    
    with nvtx.range("final_matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    
    return output

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

def benchmark_forward(model, x, warmup, n_trials=10, use_nvtx=False):
    # Warmup phase - marked with NVTX to be filtered out
    with nvtx.range("warmup"):
        for _ in range(warmup):
            with torch.no_grad():
                _ = model(x)
            cuda_synchronize()
    
    # Benchmark phase
    times = []
    for trial in range(n_trials):
        cuda_synchronize()
        
        if use_nvtx:
            with nvtx.range(f"forward_pass_trial_{trial}"):
                start = timeit.default_timer()
                with torch.no_grad():
                    _ = model(x)
                cuda_synchronize()
                end = timeit.default_timer()
        else:
            start = timeit.default_timer()
            with torch.no_grad():
                _ = model(x)
            cuda_synchronize()
            end = timeit.default_timer()
        
        times.append(end - start)
    
    return np.mean(times), np.std(times)

def benchmark_backward(model, x, warmup, n_trials=10, use_nvtx=False):
    # Warmup phase - marked with NVTX to be filtered out
    with nvtx.range("warmup"):
        for _ in range(warmup):
            model.zero_grad()
            logits = model(x)
            loss = logits.sum()
            loss.backward()
            cuda_synchronize()
    
    # Benchmark phase
    times = []
    for trial in range(n_trials):
        model.zero_grad()
        cuda_synchronize()
        
        # Forward pass (not timed)
        with nvtx.range(f"forward_for_backward_trial_{trial}"):
            logits = model(x)
            loss = logits.sum()
        
        # Time only the backward pass
        cuda_synchronize()
        
        if use_nvtx:
            with nvtx.range(f"backward_pass_trial_{trial}"):
                start = timeit.default_timer()
                loss.backward()
                cuda_synchronize()
                end = timeit.default_timer()
        else:
            start = timeit.default_timer()
            loss.backward()
            cuda_synchronize()
            end = timeit.default_timer()
        
        times.append(end - start)
    
    return np.mean(times), np.std(times)

def main():
    parser = argparse.ArgumentParser(description='Run transformer model benchmarks with NVTX profiling support')
    parser.add_argument('--warmup', type=int, default=5, help='Number of warmup rounds (default: 5)')
    parser.add_argument('--trials', type=int, default=10, help='Number of benchmark trials (default: 10)')
    parser.add_argument('--use-nvtx', action='store_true', help='Enable NVTX annotations for profiling')
    parser.add_argument('--annotate-attention', action='store_true', 
                       help='Replace attention implementation with NVTX-annotated version')
    parser.add_argument('--models', nargs='*', choices=list(model_configs.keys()), 
                       default=list(model_configs.keys()),
                       help='Models to benchmark (default: all)')
    parser.add_argument('--profile-mode', action='store_true',
                       help='Run in profile mode (single model, fewer iterations for nsys)')
    args = parser.parse_args()
    
    # If profile mode, override some settings for faster profiling
    if args.profile_mode:
        args.trials = 3  # Fewer trials for profiling
        args.use_nvtx = True  # Always use NVTX in profile mode
        if len(args.models) == len(model_configs):  # If user didn't specify models
            args.models = ['small']  # Default to just small model for profiling
    
    # Swap in annotated attention if requested
    if args.annotate_attention:
        import cs336_basics.model
        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        print("Using NVTX-annotated scaled_dot_product_attention")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running benchmarks on {device}")
    print(f"Batch size: {BATCH_SIZE}, Context length: {CONTEXT_LENGTH}")
    print(f"Warmup rounds: {args.warmup}, Benchmark trials: {args.trials}")
    if args.use_nvtx:
        print("NVTX annotations enabled")
    if args.profile_mode:
        print("Profile mode enabled (optimized for nsys profiling)")
    print()
    
    results = []
    
    for model_name in args.models:
        config = model_configs[model_name]
        
        with nvtx.range(f"benchmark_{model_name}"):
            print(f"Benchmarking {model_name} model...")
            
            # Create model
            with nvtx.range("model_creation"):
                model = create_model(**config, device=device)
                model.eval()
            
            # Create input
            x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH), device=device)
            
            # Benchmark forward pass
            with nvtx.range("forward_benchmark"):
                forward_mean, forward_std = benchmark_forward(
                    model, x, warmup=args.warmup, n_trials=args.trials, use_nvtx=args.use_nvtx
                )
            
            # Benchmark backward pass
            with nvtx.range("backward_benchmark"):
                backward_mean, backward_std = benchmark_backward(
                    model, x, warmup=args.warmup, n_trials=args.trials, use_nvtx=args.use_nvtx
                )
            
            # Store results
            results.append({
                'Model': model_name,
                'd_model': config['d_model'],
                'd_ff': config['d_ff'],
                'num_layers': config['num_layers'],
                'num_heads': config['num_heads'],
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
    if results:
        df = pd.DataFrame(results)
        print("\nSummary Results:")
        print("=" * 80)
        print(df.to_markdown(index=False, floatfmt=".4f"))
    
    if args.profile_mode:
        print("\nProfile mode complete. Run with nsys using:")
        print("nsys profile --python-backtrace=cuda --pytorch python run_benchmarks_nvtx.py --profile-mode")

if __name__ == "__main__":
    main()