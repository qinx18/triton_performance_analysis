import torch
import triton
import triton.language as tl

@triton.jit
def s151_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # s151s with k=1: a[i] = a[i] + 1 * b[i] = a[i] + b[i]
    result = a_vals + b_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s151_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s151_kernel[grid](
        a,
        b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a