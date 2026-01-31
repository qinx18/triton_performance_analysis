import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Load data with masking
    mask = indices < n
    vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Compute absolute values and find block maximum
    abs_vals = tl.abs(vals)
    
    # For masked elements, we need to ensure they don't contribute to max
    # Set masked elements to -infinity so they don't affect the maximum
    abs_vals = tl.where(mask, abs_vals, float('-inf'))
    
    block_max = tl.max(abs_vals, axis=0)
    
    # Store the block maximum
    tl.store(output_ptr + pid, block_max)

def s3113_triton(a, abs):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor for block maxima
    block_maxima = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s3113_kernel[grid](
        a, block_maxima, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction - find maximum of all block maxima
    max_val = torch.max(block_maxima)
    
    return max_val