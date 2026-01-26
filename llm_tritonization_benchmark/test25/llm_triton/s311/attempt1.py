import triton
import triton.language as tl
import torch

@triton.jit
def s311_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load data with masking
    mask = (block_start + offsets) < N
    vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Compute block sum
    block_sum = tl.sum(vals, axis=0)
    
    # Store result
    tl.store(result_ptr + pid, block_sum)

def s311_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Number of blocks needed
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Allocate output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s311_kernel[grid](a, partial_sums, N, BLOCK_SIZE)
    
    # Sum all partial results
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()