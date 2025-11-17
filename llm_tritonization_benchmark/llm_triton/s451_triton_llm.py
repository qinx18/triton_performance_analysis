import torch
import triton
import triton.language as tl

@triton.jit
def s451_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s451: a[i] = sin(b[i]) + cos(c[i])
    Uses vectorized loads and math operations for optimal GPU utilization
    """
    # Get program ID and compute element indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary conditions
    mask = offsets < n_elements
    
    # Load data with masking to handle edge cases
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute sin(b) + cos(c) using Triton's optimized math functions
    result = tl.sin(b_vals) + tl.cos(c_vals)
    
    # Store result with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def s451_triton(a, b, c):
    """
    Triton implementation of TSVC s451
    Optimized with appropriate block sizes for memory coalescing
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()  
    c = c.contiguous()
    
    n_elements = a.numel()
    
    # Use block size optimized for transcendental operations
    BLOCK_SIZE = 512
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s451_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a