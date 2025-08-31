"""
Memory profiling script for transformer models.

This script profiles memory usage of the 2.7B model with different context lengths
and supports both forward-only and full training step profiling.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from cs336_basics.model import BasicsTransformerLM
import argparse
import gc
import time
from typing import Dict, Optional
import pandas as pd

# Model configuration for 2.7B model
MODEL_2_7B_CONFIG = {
    'd_model': 2560,
    'd_ff': 10240,
    'num_layers': 32,
    'num_heads': 32,
}

VOCAB_SIZE = 10000
BATCH_SIZE = 4
WARMUP_STEPS = 3

def cuda_synchronize():
    """Synchronize CUDA if available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def get_memory_stats() -> Dict[str, float]:
    """Get current memory statistics in MB."""
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'peak': 0}
    
    return {
        'allocated': torch.cuda.memory_allocated() / (1024 * 1024),
        'reserved': torch.cuda.memory_reserved() / (1024 * 1024),
        'peak': torch.cuda.max_memory_allocated() / (1024 * 1024)
    }

def create_model(context_length: int, device: str = 'cuda') -> BasicsTransformerLM:
    """Create the 2.7B model with specified context length."""
    return BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=MODEL_2_7B_CONFIG['d_model'],
        num_layers=MODEL_2_7B_CONFIG['num_layers'],
        num_heads=MODEL_2_7B_CONFIG['num_heads'],
        d_ff=MODEL_2_7B_CONFIG['d_ff'],
        rope_theta=10000
    ).to(device=device)

def profile_forward_pass(
    model: nn.Module,
    context_length: int,
    device: str = 'cuda',
    use_amp: bool = False,
    profile_memory: bool = False,
    snapshot_name: Optional[str] = None
) -> Dict[str, float]:
    """Profile forward pass only."""
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Create input
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device)
    
    # Warmup
    print(f"  Running {WARMUP_STEPS} warmup steps...")
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            if use_amp:
                with autocast():
                    _ = model(x)
            else:
                _ = model(x)
        cuda_synchronize()
    
    # Clear memory stats after warmup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Start memory profiling if requested
    if profile_memory and torch.cuda.is_available():
        print("  Starting memory profiling...")
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    # Profile forward pass
    print("  Running forward pass...")
    cuda_synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        if use_amp:
            with autocast():
                output = model(x)
        else:
            output = model(x)
    
    cuda_synchronize()
    forward_time = time.time() - start_time
    
    # Save memory snapshot if requested
    if profile_memory and torch.cuda.is_available() and snapshot_name:
        print(f"  Saving memory snapshot to {snapshot_name}...")
        torch.cuda.memory._dump_snapshot(snapshot_name)
        torch.cuda.memory._record_memory_history(enabled=None)
    
    # Get memory stats
    memory_stats = get_memory_stats()
    
    # Clean up
    del output, x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'forward_time': forward_time * 1000,  # Convert to ms
        'peak_memory_mb': memory_stats['peak'],
        'allocated_memory_mb': memory_stats['allocated']
    }

def profile_training_step(
    model: nn.Module,
    context_length: int,
    device: str = 'cuda',
    use_amp: bool = False,
    profile_memory: bool = False,
    snapshot_name: Optional[str] = None
) -> Dict[str, float]:
    """Profile full training step (forward + backward + optimizer)."""
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler() if use_amp else None
    
    # Create input
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device)
    y = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, context_length), device=device)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Warmup
    print(f"  Running {WARMUP_STEPS} warmup steps...")
    for _ in range(WARMUP_STEPS):
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                logits = model(x)
                logits_flat = logits.view(-1, VOCAB_SIZE)
                y_flat = y.view(-1)
                loss = criterion(logits_flat, y_flat)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            logits_flat = logits.view(-1, VOCAB_SIZE)
            y_flat = y.view(-1)
            loss = criterion(logits_flat, y_flat)
            loss.backward()
            optimizer.step()
        
        cuda_synchronize()
    
    # Clear memory stats after warmup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Start memory profiling if requested
    if profile_memory and torch.cuda.is_available():
        print("  Starting memory profiling...")
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    # Profile training step
    print("  Running training step (forward + backward + optimizer)...")
    optimizer.zero_grad()
    cuda_synchronize()
    start_time = time.time()
    
    # Forward pass
    if use_amp:
        with autocast():
            logits = model(x)
            logits_flat = logits.view(-1, VOCAB_SIZE)
            y_flat = y.view(-1)
            loss = criterion(logits_flat, y_flat)
    else:
        logits = model(x)
        logits_flat = logits.view(-1, VOCAB_SIZE)
        y_flat = y.view(-1)
        loss = criterion(logits_flat, y_flat)
    
    cuda_synchronize()
    forward_time = time.time() - start_time
    
    # Backward pass
    backward_start = time.time()
    if use_amp:
        scaler.scale(loss).backward()
    else:
        loss.backward()
    cuda_synchronize()
    backward_time = time.time() - backward_start
    
    # Optimizer step
    optimizer_start = time.time()
    if use_amp:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    cuda_synchronize()
    optimizer_time = time.time() - optimizer_start
    
    total_time = time.time() - start_time
    
    # Save memory snapshot if requested
    if profile_memory and torch.cuda.is_available() and snapshot_name:
        print(f"  Saving memory snapshot to {snapshot_name}...")
        torch.cuda.memory._dump_snapshot(snapshot_name)
        torch.cuda.memory._record_memory_history(enabled=None)
    
    # Get memory stats
    memory_stats = get_memory_stats()
    
    # Clean up
    del logits, loss, x, y
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return {
        'forward_time': forward_time * 1000,  # Convert to ms
        'backward_time': backward_time * 1000,
        'optimizer_time': optimizer_time * 1000,
        'total_time': total_time * 1000,
        'peak_memory_mb': memory_stats['peak'],
        'allocated_memory_mb': memory_stats['allocated']
    }

def calculate_activation_size(context_length: int) -> float:
    """
    Calculate the size of a tensor of activations in the Transformer residual stream.
    
    For 2.7B model:
    - d_model = 2560
    - Single precision (float32) = 4 bytes per element
    
    Residual stream shape: (batch_size, context_length, d_model)
    """
    d_model = MODEL_2_7B_CONFIG['d_model']
    batch_size = BATCH_SIZE
    bytes_per_float32 = 4
    
    # Total elements in residual stream tensor
    total_elements = batch_size * context_length * d_model
    
    # Size in bytes
    size_bytes = total_elements * bytes_per_float32
    
    # Convert to MB
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb

def main():
    parser = argparse.ArgumentParser(description='Memory profiling for 2.7B transformer model')
    parser.add_argument('--mode', choices=['forward', 'training'], default='forward',
                       help='Profiling mode: forward pass only or full training step')
    parser.add_argument('--context-lengths', nargs='+', type=int, default=[128, 256, 512],
                       help='Context lengths to profile')
    parser.add_argument('--use-amp', action='store_true',
                       help='Use automatic mixed precision (AMP)')
    parser.add_argument('--profile-memory', action='store_true',
                       help='Enable detailed memory profiling with snapshots')
    parser.add_argument('--snapshot-prefix', default='memory_snapshot',
                       help='Prefix for memory snapshot files')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("Memory Profiling for 2.7B Model")
    print("=" * 80)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_memory:.2f} GB")
    print(f"Mode: {args.mode}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Mixed precision: {args.use_amp}")
    print(f"Memory profiling: {args.profile_memory}")
    print("=" * 80)
    print()
    
    results = []
    
    for context_length in args.context_lengths:
        print(f"Profiling context_length={context_length}:")
        
        try:
            # Create model
            print("  Creating model...")
            model = create_model(context_length, device)
            
            # Generate snapshot name if profiling
            snapshot_name = None
            if args.profile_memory:
                mode_str = 'forward' if args.mode == 'forward' else 'training'
                amp_str = '_amp' if args.use_amp else ''
                snapshot_name = f"{args.snapshot_prefix}_{mode_str}_ctx{context_length}{amp_str}.pickle"
            
            # Profile based on mode
            if args.mode == 'forward':
                stats = profile_forward_pass(
                    model, context_length, device, 
                    use_amp=args.use_amp,
                    profile_memory=args.profile_memory,
                    snapshot_name=snapshot_name
                )
            else:  # training
                stats = profile_training_step(
                    model, context_length, device,
                    use_amp=args.use_amp,
                    profile_memory=args.profile_memory,
                    snapshot_name=snapshot_name
                )
            
            # Add context length and mode to stats
            stats['context_length'] = context_length
            stats['mode'] = args.mode
            stats['use_amp'] = args.use_amp
            
            # Calculate activation size
            activation_size = calculate_activation_size(context_length)
            stats['activation_size_mb'] = activation_size
            
            results.append(stats)
            
            # Print results
            print(f"  ✓ Peak memory: {stats['peak_memory_mb']:.1f} MB")
            if args.mode == 'forward':
                print(f"  ✓ Forward time: {stats['forward_time']:.2f} ms")
            else:
                print(f"  ✓ Forward time: {stats['forward_time']:.2f} ms")
                print(f"  ✓ Backward time: {stats['backward_time']:.2f} ms")
                print(f"  ✓ Optimizer time: {stats['optimizer_time']:.2f} ms")
                print(f"  ✓ Total time: {stats['total_time']:.2f} ms")
            print(f"  ✓ Residual stream activation size: {activation_size:.2f} MB")
            
            if snapshot_name and args.profile_memory:
                print(f"  ✓ Memory snapshot saved to: {snapshot_name}")
            
            # Clean up
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except torch.cuda.OutOfMemoryError:
            print(f"  ✗ Out of Memory!")
            results.append({
                'context_length': context_length,
                'mode': args.mode,
                'use_amp': args.use_amp,
                'status': 'OOM'
            })
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'context_length': context_length,
                'mode': args.mode,
                'use_amp': args.use_amp,
                'status': 'error',
                'error': str(e)
            })
        
        print()
    
    # Display summary table
    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("SUMMARY RESULTS")
        print("=" * 80)
        
        # Filter successful runs
        successful = [r for r in results if 'peak_memory_mb' in r]
        if successful:
            summary_df = pd.DataFrame(successful)
            
            print("\nMemory Usage Summary:")
            print("-" * 40)
            display_cols = ['context_length', 'mode', 'use_amp', 'peak_memory_mb', 'activation_size_mb']
            if args.mode == 'training':
                display_cols.extend(['forward_time', 'backward_time', 'optimizer_time', 'total_time'])
            else:
                display_cols.append('forward_time')
            
            print(summary_df[display_cols].to_string(index=False, float_format='%.2f'))
            
            # Save to CSV
            output_file = f'memory_profile_results_{args.mode}.csv'
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        
        # Show instructions for viewing snapshots
        if args.profile_memory:
            print("\n" + "=" * 80)
            print("VIEWING MEMORY SNAPSHOTS")
            print("=" * 80)
            print("To view the memory snapshots:")
            print("1. Open https://pytorch.org/memory_viz in a web browser")
            print("2. Drag and drop the .pickle files onto the page")
            print("\nGenerated snapshot files:")
            for r in successful:
                if 'context_length' in r:
                    mode_str = 'forward' if r['mode'] == 'forward' else 'training'
                    amp_str = '_amp' if r.get('use_amp', False) else ''
                    fname = f"{args.snapshot_prefix}_{mode_str}_ctx{r['context_length']}{amp_str}.pickle"
                    print(f"  - {fname}")

if __name__ == "__main__":
    main()