import torch
import triton
import triton.language as tl

@triton.jit
def s316_kernel(a_ptr, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for finding minimum value using block-wise reduction
    """
    # Get program ID and compute block start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load data with masking, use large value for masked elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=float('inf'))
    
    # Perform block-wise minimum reduction
    min_val = tl.min(a_vals)
    
    # Store result for each block (will be reduced further on CPU)
    tl.store(x_ptr + pid, min_val)

def s316_triton(a, x):
    """
    Triton implementation of s316 - finding minimum value
    Uses block-wise reduction followed by CPU reduction for final result
    """
    a = a.contiguous()
    x = x.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary buffer for block results
    block_mins = torch.empty(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel with appropriate grid
    s316_kernel[(grid_size,)](
        a, block_mins, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction on CPU/GPU depending on size
    if grid_size > 1:
        final_min = torch.min(block_mins)
    else:
        final_min = block_mins[0]
    
    # Store result in output tensor
    x[0] = final_min
    
    return x