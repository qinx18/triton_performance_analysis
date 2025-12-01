import torch
import triton
import triton.language as tl

@triton.jit
def s1351_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s1351 - vectorized linear combination with coalesced memory access
    """
    # Calculate block start position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for handling array bounds
    mask = offsets < n_elements
    
    # Load input vectors with masking
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute linear combination
    result = b + c
    
    # Store result with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def s1351_triton(a, b, c):
    """
    Triton implementation of TSVC s1351 - basic linear combination.
    Optimized with vectorized operations and coalesced memory access.
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    n_elements = a.numel()
    
    # Use block size of 1024 for good occupancy on most GPUs
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with flattened tensor views
    s1351_kernel[grid](
        a.view(-1), b.view(-1), c.view(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a