import torch
import triton
import triton.language as tl

@triton.jit
def vpv_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary conditions
    mask = offsets < n_elements
    
    # Load data with masking
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Perform vectorized addition
    result = a + b
    
    # Store result back to a with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def vpv_triton(a, b):
    """
    Triton implementation of TSVC vpv function.
    Performs element-wise addition: a[i] += b[i]
    """
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    vpv_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a