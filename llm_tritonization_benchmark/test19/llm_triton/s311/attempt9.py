import triton
import triton.language as tl
import torch

@triton.jit
def s311_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel computes sum reduction of array a
    pid = tl.program_id(0)
    
    # Compute block boundaries
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load data with masking
    mask = block_start + offsets < n
    vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Compute partial sum for this block
    partial_sum = tl.sum(vals, axis=0)
    
    # Store partial result
    tl.store(result_ptr + pid, partial_sum)

def s311_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s311_kernel[grid](a, partial_sums, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Final reduction
    total_sum = torch.sum(partial_sums)
    
    # Calculate checksum - use abs of sum
    chksum = torch.abs(total_sum)
    
    # Return the checksum value as float
    return chksum.item()