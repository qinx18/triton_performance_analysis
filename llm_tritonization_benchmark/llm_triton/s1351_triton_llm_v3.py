import torch
import triton
import triton.language as tl

@triton.jit
def s1351_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load B and C values
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    
    # Compute A = B + C
    a_vals = b_vals + c_vals
    
    # Store result
    tl.store(a_ptr + block_start + offsets, a_vals, mask=mask)

def s1351_triton(a, b, c):
    n_elements = a.numel()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1351_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a