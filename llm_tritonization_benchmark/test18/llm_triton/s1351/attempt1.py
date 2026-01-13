import torch
import triton
import triton.language as tl

@triton.jit
def s1351_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    result = b_vals + c_vals
    
    tl.store(a_ptr + idx, result, mask=mask)

def s1351_triton(a, b, c):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1351_kernel[grid](
        a, b, c,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a