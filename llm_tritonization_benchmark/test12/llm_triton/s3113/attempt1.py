import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = current_offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute absolute values
    abs_vals = tl.abs(a_vals)
    
    # Find maximum in this block
    block_max = tl.max(abs_vals, axis=0)
    
    # Store the block maximum
    tl.store(output_ptr + pid, block_max)

def s3113_triton(a, abs):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for block maxima
    block_maxima = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s3113_kernel[(num_blocks,)](
        a, block_maxima, n_elements, BLOCK_SIZE
    )
    
    # Find global maximum among block maxima
    max_val = torch.max(block_maxima)
    
    return max_val