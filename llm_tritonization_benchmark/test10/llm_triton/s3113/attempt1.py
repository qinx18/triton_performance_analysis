import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Compute absolute values
    abs_vals = tl.abs(a_vals)
    
    # Find maximum in this block
    block_max = tl.max(abs_vals)
    
    # Store the block maximum
    tl.store(output_ptr + pid, block_max)

def s3113_triton(a, abs):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for block maximums
    block_maxs = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s3113_kernel[(num_blocks,)](
        a, block_maxs, n_elements, BLOCK_SIZE
    )
    
    # Find global maximum from block maximums
    max_val = torch.max(block_maxs)
    
    return max_val