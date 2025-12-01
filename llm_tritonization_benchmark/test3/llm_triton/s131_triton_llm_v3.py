import triton
import triton.language as tl
import torch

@triton.jit
def s131_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Read from copy for a[i + 1] values
    a_vals = tl.load(a_copy_ptr + offsets + 1, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    result = a_vals + b_vals
    
    # Write to original array
    tl.store(a_ptr + offsets, result, mask=mask)

def s131_triton(a, b):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        BLOCK_SIZE,
    )