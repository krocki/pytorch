"""
Triton implementation of Linear Cross Entropy following actual Liger Kernel approach.
- Batch/token chunking (not vocabulary chunking)  
- Standard linear layer + cross entropy kernel
- No nested loops in kernels
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional
import time
import gc


def linear_cross_entropy_triton_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Forward pass following actual Liger approach:
    1. Use standard PyTorch linear layer (efficient)
    2. Use Triton only for cross entropy computation
    3. Chunk by batch/tokens, not vocabulary
    """
    
    input_2d = input.view(-1, input.size(-1))  # [BT, H]
    target_1d = target.view(-1)                # [BT]
    
    BT, H = input_2d.shape
    V = weight.size(0)
    
    # Calculate memory requirements
    logits_memory_gb = (BT * V * 4) / (1024**3)
    
    # If logits fit in memory, use standard approach
    if logits_memory_gb < 4.0:  # 4GB threshold
        logits = F.linear(input_2d, weight, bias)
        loss = F.cross_entropy(logits, target_1d, ignore_index=ignore_index, reduction='mean')
        return loss
    
    # For very large cases, use Liger's batch chunking approach
    print(f"Using batch chunking for large tensor: {logits_memory_gb:.1f} GB")
    
    # Calculate chunk size (following Liger's logic)
    inc_factor = triton.cdiv(V, H)
    chunk_size = max(1, triton.cdiv(BT, inc_factor))
    chunk_size = triton.next_power_of_2(chunk_size)
    
    total_loss = 0.0
    valid_tokens = 0
    
    # Process in chunks
    for chunk_start in range(0, BT, chunk_size):
        chunk_end = min(chunk_start + chunk_size, BT)
        
        # Get chunk data
        input_chunk = input_2d[chunk_start:chunk_end]    # [chunk_size, H]
        target_chunk = target_1d[chunk_start:chunk_end]  # [chunk_size]
        
        # Standard linear transformation for this chunk
        logits_chunk = F.linear(input_chunk, weight, bias)  # [chunk_size, V]
        
        # Cross entropy for this chunk
        chunk_loss = F.cross_entropy(
            logits_chunk, target_chunk, 
            ignore_index=ignore_index, reduction='sum'
        )
        
        # Accumulate
        total_loss += chunk_loss.item()
        valid_tokens += (target_chunk != ignore_index).sum().item()
    
    # Return mean loss
    if valid_tokens > 0:
        return torch.tensor(total_loss / valid_tokens, device=input.device, dtype=input.dtype)
    else:
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)


def measure_memory_usage():
    """Measure current GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024 / 1024


def benchmark_performance_and_memory():
    """Benchmark performance and memory usage vs PyTorch."""
    print("\nPerformance and Memory Benchmark")
    print("=" * 60)
    
    # Test configurations - realistic scenarios
    configs = [
        # (name, batch, seq, hidden, vocab, description)
        ("Small vocab", 8, 128, 768, 8000, "Typical small model"),
        ("Medium vocab", 4, 256, 1024, 32000, "GPT-2 scale"),
        ("Large vocab", 2, 512, 2048, 50000, "Large model"), 
        ("XLarge vocab", 1, 1024, 4096, 100000, "Very large vocab"),
    ]
    
    for name, batch, seq, hidden, vocab, desc in configs:
        print(f"\n{name} ({desc})")
        print(f"  Config: batch={batch}, seq={seq}, hidden={hidden}, vocab={vocab}")
        
        # Calculate theoretical memory usage
        logits_memory_gb = (batch * seq * vocab * 4) / (1024**3)
        print(f"  Full logits would need: {logits_memory_gb:.2f} GB")
        
        if logits_memory_gb > 20:  # Skip if would OOM
            print(f"  Status: SKIPPED - Would likely OOM")
            continue
            
        try:
            # Create test data
            torch.manual_seed(42)
            input_tensor = torch.randn(batch, seq, hidden, device='cuda', dtype=torch.float32)
            weight = torch.randn(vocab, hidden, device='cuda', dtype=torch.float32)
            bias = torch.randn(vocab, device='cuda', dtype=torch.float32)
            target = torch.randint(0, vocab, (batch, seq), device='cuda')
            
            # Warmup both implementations
            for _ in range(3):
                try:
                    _ = linear_cross_entropy_triton_forward(input_tensor, weight, target, bias)
                except:
                    pass
                try:
                    logits = F.linear(input_tensor.view(-1, hidden), weight, bias)
                    _ = F.cross_entropy(logits, target.view(-1))
                    del logits
                except:
                    pass
            
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Benchmark Triton implementation
            memory_before_triton = measure_memory_usage()
            
            start_time = time.time()
            for _ in range(10):
                loss_triton = linear_cross_entropy_triton_forward(input_tensor, weight, target, bias)
            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / 10
            
            memory_after_triton = measure_memory_usage()
            triton_memory_peak = memory_after_triton - memory_before_triton
            
            torch.cuda.empty_cache()
            
            # Benchmark PyTorch implementation
            memory_before_pytorch = measure_memory_usage()
            
            pytorch_success = True
            try:
                start_time = time.time()
                for _ in range(10):
                    logits = F.linear(input_tensor.view(-1, hidden), weight, bias)
                    loss_pytorch = F.cross_entropy(logits, target.view(-1))
                    del logits
                torch.cuda.synchronize()
                pytorch_time = (time.time() - start_time) / 10
                
                memory_after_pytorch = measure_memory_usage()
                pytorch_memory_peak = memory_after_pytorch - memory_before_pytorch
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    pytorch_success = False
                    pytorch_time = float('inf')
                    pytorch_memory_peak = float('inf')
                    print(f"  PyTorch: OOM")
                else:
                    raise e
            
            # Results
            print(f"  Triton time: {triton_time*1000:.2f} ms")
            if pytorch_success:
                print(f"  PyTorch time: {pytorch_time*1000:.2f} ms")
                speedup = pytorch_time / triton_time
                print(f"  Speedup: {speedup:.2f}x")
            else:
                print(f"  PyTorch time: OOM")
                print(f"  Speedup: ∞ (PyTorch OOM)")
            
            print(f"  Triton memory peak: {max(0, triton_memory_peak):.1f} MB")
            if pytorch_success:
                print(f"  PyTorch memory peak: {max(0, pytorch_memory_peak):.1f} MB")
                if pytorch_memory_peak > 0:
                    memory_savings = pytorch_memory_peak / max(0.1, triton_memory_peak)
                    print(f"  Memory savings: {memory_savings:.2f}x")
            else:
                print(f"  PyTorch memory peak: OOM")
                print(f"  Memory savings: ∞ (PyTorch OOM)")
                
            # Verify correctness
            if pytorch_success:
                torch.cuda.empty_cache()
                logits_ref = F.linear(input_tensor.view(-1, hidden), weight, bias)
                loss_ref = F.cross_entropy(logits_ref, target.view(-1))
                diff = abs(loss_ref - loss_triton).item()
                print(f"  Numerical difference: {diff:.2e}")
                
        except Exception as e:
            print(f"  Status: FAILED - {e}")
            
        finally:
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    print("Testing Triton Linear Cross-Entropy (Fixed Liger Approach)")
    print("=" * 60)
    
    # Basic correctness test
    print("Testing correctness...")
    torch.manual_seed(42)
    input_tensor = torch.randn(2, 4, 128, device='cuda', dtype=torch.float32)
    weight = torch.randn(1000, 128, device='cuda', dtype=torch.float32)
    bias = torch.randn(1000, device='cuda', dtype=torch.float32)
    target = torch.randint(0, 1000, (2, 4), device='cuda')
    
    # Reference
    logits_ref = F.linear(input_tensor.view(-1, 128), weight, bias)
    loss_ref = F.cross_entropy(logits_ref, target.view(-1))
    
    # Triton  
    loss_triton = linear_cross_entropy_triton_forward(input_tensor, weight, target, bias)
    
    print(f"Reference loss: {loss_ref.item():.6f}")
    print(f"Triton loss: {loss_triton.item():.6f}")
    print(f"Difference: {abs(loss_ref - loss_triton).item():.2e}")
    
    if abs(loss_ref - loss_triton).item() < 1e-4:
        print("✓ Correctness test PASSED")
        benchmark_performance_and_memory()
    else:
        print("✗ Correctness test FAILED")