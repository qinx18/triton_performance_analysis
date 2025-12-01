import torch
import triton
import triton.language as tl

@triton.jit
def s315_kernel(
    a_ptr,
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to find maximum value in array.
    Uses block-level reduction for efficient parallel max finding.
    """
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load data with masking, use -inf for invalid elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find maximum value in this block
    block_max = tl.max(a_vals, axis=0)
    
    # Store block maximum (will be reduced later on CPU)
    if pid == 0:
        tl.store(x_ptr, block_max)
    else:
        # For multi-block case, use atomic max
        tl.atomic_max(x_ptr, block_max)

def s315_triton(a, x):
    """
    Triton implementation of TSVC s315 - find maximum value and its index.
    Uses GPU parallel reduction for efficient max finding.
    """
    a = a.contiguous()
    x = x.contiguous()
    
    n_elements = a.numel()
    
    # For small arrays or single element, use PyTorch directly
    if n_elements <= 1024:
        max_val, max_idx = torch.max(a, dim=0)
        x[0] = max_val
        return a, x
    
    # Choose block size based on array size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # Create temporary buffer for intermediate results
    temp_max = torch.full((1,), float('-inf'), device=a.device, dtype=a.dtype)
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel for parallel max reduction
    s315_kernel[grid](
        a, temp_max, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # For multiple blocks, we need a second pass or use PyTorch for final reduction
    if grid[0] > 1:
        # Fall back to PyTorch for final max (atomic_max handles this in kernel)
        max_val = temp_max[0]
    else:
        max_val = temp_max[0]
    
    # Update x with the maximum value
    x[0] = max_val
    
    return a, x