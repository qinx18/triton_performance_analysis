import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(
    a_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Load block of elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements and compute absolute values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(a_vals)
    
    # Find maximum within this block
    block_max = tl.max(abs_vals, axis=0)
    
    # Store block maximum
    tl.store(output_ptr + pid, block_max)

def s3113_triton(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for block maxima
    block_maxima = torch.empty(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel to compute block maxima
    grid = (n_blocks,)
    s3113_kernel[grid](
        a,
        block_maxima,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Find global maximum from block maxima on CPU for final reduction
    max_val = torch.max(block_maxima).item()
    
    return max_val