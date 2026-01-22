import triton
import triton.language as tl
import torch

@triton.jit
def s311_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Calculate which block this program handles
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    
    # Load data with masking for edge cases
    mask = (block_start + offsets) < n
    vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Compute block sum
    block_sum = tl.sum(vals, axis=0)
    
    # Store the partial sum
    tl.store(result_ptr + pid, block_sum)

def s311_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Number of blocks needed
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Allocate output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s311_kernel[grid](a, partial_sums, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Final reduction on CPU/GPU
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()