import torch
import triton
import triton.language as tl

@triton.jit
def vpvtv_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate starting position for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle cases where n_elements is not divisible by BLOCK_SIZE
    mask = offsets < n_elements
    
    # Load data with masking to handle edge cases
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    a = tl.load(a_ptr + offsets, mask=mask)
    
    # Perform vectorized multiply-add: a += b * c
    result = a + b * c
    
    # Store result back to memory
    tl.store(a_ptr + offsets, result, mask=mask)

def vpvtv_triton(a, b, c):
    """
    Triton implementation of TSVC vpvtv function.
    Optimized for GPU with vectorized operations and coalesced memory access.
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous() 
    c = c.contiguous()
    
    n_elements = a.numel()
    
    # Use 1024 block size for good occupancy on most GPUs
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with optimized block size
    vpvtv_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a