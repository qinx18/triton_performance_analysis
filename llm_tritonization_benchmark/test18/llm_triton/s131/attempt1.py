import triton
import triton.language as tl
import torch

@triton.jit
def s131_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    m,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    a_copy_vals = tl.load(a_copy_ptr + idx + m, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    result = a_copy_vals + b_vals
    
    tl.store(a_ptr + idx, result, mask=mask)

def s131_triton(a, b, m):
    N = a.shape[0]
    n_elements = N - 1
    
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        m,
        BLOCK_SIZE=BLOCK_SIZE,
    )