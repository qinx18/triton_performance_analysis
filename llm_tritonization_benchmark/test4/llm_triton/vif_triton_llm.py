import torch
import triton
import triton.language as tl

@triton.jit
def vif_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for vectorized conditional assignment.
    Optimized with coalesced memory access and efficient masking.
    """
    # Calculate block start position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements to handle edge cases
    mask = offsets < n_elements
    
    # Load data with masking for out-of-bounds protection
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Vectorized conditional: if b > 0, use b, otherwise keep a
    condition = b_vals > 0.0
    result = tl.where(condition, b_vals, a_vals)
    
    # Store result back to a with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def vif_triton(a, b):
    """
    Triton implementation of TSVC vif function with optimized GPU execution.
    Uses block-wise processing for memory coalescing and efficient parallelization.
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    
    # Optimized block size for memory coalescing and occupancy
    BLOCK_SIZE = 1024
    
    # Calculate grid size for complete coverage
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with flattened tensor views
    vif_kernel[grid](
        a.view(-1), 
        b.view(-1), 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a