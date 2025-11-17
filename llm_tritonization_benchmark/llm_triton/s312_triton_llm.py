import torch
import triton
import triton.language as tl

@triton.jit
def s312_kernel(a_ptr, prod_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for computing product of array elements using reduction.
    Uses shared memory reduction within each block for efficiency.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data with masking for edge cases
    mask = offsets < n_elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=1.0)
    
    # Compute product within block using reduction
    block_prod = tl.reduce(a_vals, axis=0, op=tl.mul)
    
    # Store partial product (first thread in block writes result)
    if tl.program_id(axis=0) == 0:
        tl.store(prod_ptr, block_prod)

def s312_triton(a):
    """
    Triton implementation of TSVC s312 - product of array elements.
    Uses GPU reduction for efficient parallel computation.
    """
    a = a.contiguous()
    n_elements = a.numel()
    
    # Handle empty tensor edge case
    if n_elements == 0:
        return a
    
    # Create output tensor for product result
    prod = torch.ones(1, dtype=a.dtype, device=a.device)
    
    # Use optimal block size for reduction
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # For multiple blocks, need multi-stage reduction
    if grid[0] > 1:
        # First stage: reduce within blocks
        partial_prods = torch.ones(grid[0], dtype=a.dtype, device=a.device)
        
        @triton.jit
        def s312_first_stage(a_ptr, partial_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            
            mask = offsets < n_elements
            a_vals = tl.load(a_ptr + offsets, mask=mask, other=1.0)
            block_prod = tl.reduce(a_vals, axis=0, op=tl.mul)
            
            tl.store(partial_ptr + pid, block_prod)
        
        s312_first_stage[grid](a, partial_prods, n_elements, BLOCK_SIZE)
        
        # Second stage: reduce partial products on CPU (small array)
        prod = torch.prod(partial_prods).unsqueeze(0)
    else:
        # Single block case
        s312_kernel[grid](a, prod, n_elements, BLOCK_SIZE)
    
    return a