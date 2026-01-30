import triton
import triton.language as tl
import torch

@triton.jit
def s131_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    m,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Read from copy and b
    a_vals = tl.load(a_copy_ptr + indices + m, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + indices, result, mask=mask)

def s131_triton(a, b, m):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a,
        a_copy,
        b,
        m,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )