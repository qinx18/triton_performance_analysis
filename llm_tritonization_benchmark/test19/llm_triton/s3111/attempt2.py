import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Apply condition: a[i] > 0
    condition = a_vals > 0.0
    conditional_vals = tl.where(condition, a_vals, 0.0)
    
    # Sum within this block
    block_sum = tl.sum(conditional_vals, axis=0)
    
    # Store partial result
    tl.store(result_ptr + pid, block_sum)

def s3111_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor for partial results
    partial_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s3111_kernel[grid](
        a, partial_results, n, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction on CPU
    return partial_results.sum().item()