import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(
    a_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s116 - processes 5 consecutive elements with dependencies.
    Uses block-level processing with proper dependency handling.
    """
    # Get program ID and calculate starting position (aligned to groups of 5)
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate offsets for the current block (process in groups of 5)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    # Need to ensure we don't go beyond n_elements - 5
    mask = offsets < (n_elements - 5)
    
    # Only process if we have valid indices and they're aligned to groups of 5
    valid_mask = mask & ((offsets % 5) == 0)
    
    # Load groups of 6 consecutive elements (5 + 1 for dependency)
    base_offsets = tl.where(valid_mask, offsets, 0)
    
    # Load 6 consecutive values starting from each base offset
    a0 = tl.load(a_ptr + base_offsets, mask=valid_mask, other=0.0)
    a1 = tl.load(a_ptr + base_offsets + 1, mask=valid_mask, other=0.0)
    a2 = tl.load(a_ptr + base_offsets + 2, mask=valid_mask, other=0.0)
    a3 = tl.load(a_ptr + base_offsets + 3, mask=valid_mask, other=0.0)
    a4 = tl.load(a_ptr + base_offsets + 4, mask=valid_mask, other=0.0)
    a5 = tl.load(a_ptr + base_offsets + 5, mask=valid_mask, other=0.0)
    
    # Compute the 5 operations with proper dependencies
    # a[i] = a[i + 1] * a[i]
    # a[i + 1] = a[i + 2] * a[i + 1]
    # a[i + 2] = a[i + 3] * a[i + 2]
    # a[i + 3] = a[i + 4] * a[i + 3]
    # a[i + 4] = a[i + 5] * a[i + 4]
    
    new_a0 = a1 * a0
    new_a1 = a2 * a1
    new_a2 = a3 * a2
    new_a3 = a4 * a3
    new_a4 = a5 * a4
    
    # Store results back to memory
    tl.store(a_ptr + base_offsets, new_a0, mask=valid_mask)
    tl.store(a_ptr + base_offsets + 1, new_a1, mask=valid_mask)
    tl.store(a_ptr + base_offsets + 2, new_a2, mask=valid_mask)
    tl.store(a_ptr + base_offsets + 3, new_a3, mask=valid_mask)
    tl.store(a_ptr + base_offsets + 4, new_a4, mask=valid_mask)


def s116_triton(a):
    """
    Triton implementation of TSVC s116 - linear dependence testing with unrolling.
    Optimized for GPU execution with proper memory coalescing and dependency handling.
    
    Args:
        a: Input/output tensor
        
    Returns:
        torch.Tensor: Modified array a
    """
    a = a.contiguous()
    n = a.size(0)
    
    if n <= 5:
        return a
    
    # Calculate number of groups of 5 elements we can process
    n_groups = (n - 5) // 5
    if n_groups == 0:
        return a
    
    # Use block size that processes multiple groups efficiently
    BLOCK_SIZE = 64  # Process 64 groups of 5 at once
    
    # Calculate grid size - number of blocks needed
    grid_size = triton.cdiv(n_groups, BLOCK_SIZE)
    
    # Launch kernel with appropriate grid
    s116_kernel[(grid_size,)](
        a,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a