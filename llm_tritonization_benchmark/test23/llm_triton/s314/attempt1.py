import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Each block computes the max of a portion of the array
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Load data with masking
    mask = indices < n
    vals = tl.load(a_ptr + indices, mask=mask, other=float('-inf'))
    
    # Compute block maximum
    block_max = tl.max(vals, axis=0)
    
    # Store the block maximum
    tl.store(result_ptr + tl.program_id(0), block_max)

def s314_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Number of blocks needed
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor for block maximums
    block_maxs = torch.empty(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s314_kernel[grid](
        a, block_maxs, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Find the overall maximum from block maximums
    x = torch.max(block_maxs).item()
    
    return x