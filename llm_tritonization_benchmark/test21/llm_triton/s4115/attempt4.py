import triton
import triton.language as tl
import torch

@triton.jit
def s4115_kernel(a_ptr, b_ptr, ip_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    partial_sum = tl.zeros(1, dtype=tl.float32)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load a[i]
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load indices ip[i] and clamp to valid range
        indices = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load b[ip[i]] - gather operation
        b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
        
        # Compute a[i] * b[ip[i]]
        products = a_vals * b_vals
        
        # Accumulate to partial sum
        partial_sum += tl.sum(products, axis=0)
    
    # Each work item writes its partial sum
    pid = tl.program_id(0)
    tl.store(result_ptr + pid, partial_sum)

def s4115_triton(a, b, ip):
    n = a.shape[0]
    
    # Use multiple blocks for parallel reduction
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    grid = (num_blocks,)
    
    # Create result tensor for partial sums
    partial_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    s4115_kernel[grid](
        a, b, ip, partial_results,
        n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction on CPU
    return partial_results.sum().item()