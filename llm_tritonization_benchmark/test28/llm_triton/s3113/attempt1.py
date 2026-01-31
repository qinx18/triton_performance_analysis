import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load data with masking
    mask = offsets < n
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Compute absolute values and find block maximum
    abs_vals = tl.abs(vals)
    block_max = tl.max(abs_vals, axis=0)
    
    # Store the block maximum
    pid = tl.program_id(0)
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
    
    # Final reduction on CPU/GPU
    max_val = torch.max(block_maxima)
    
    return max_val