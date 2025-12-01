import torch
import triton
import triton.language as tl

@triton.jit
def s151_kernel(
    a_ptr, b_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offsets < n_elements
    
    # Load data with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Perform computation: a + b + 1.0
    result = a_vals + b_vals + 1.0
    
    # Store result with masking
    tl.store(output_ptr + offsets, result, mask=mask)

def s151_triton(a, b):
    """
    Triton-optimized version of s151 - Scalar and array expansion
    Uses vectorized operations with optimal block sizes for GPU acceleration
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    
    # Create output tensor
    result_a = torch.empty_like(a)
    
    # Choose block size - power of 2 for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s151_kernel[grid](
        a, b, result_a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result_a