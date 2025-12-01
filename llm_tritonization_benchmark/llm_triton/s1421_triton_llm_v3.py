import torch
import triton
import triton.language as tl

@triton.jit
def s1421_kernel(
    a_ptr, b_ptr, xx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    xx_vals = tl.load(xx_ptr + offsets, mask=mask)
    
    result = xx_vals + a_vals
    
    tl.store(b_ptr + offsets, result, mask=mask)

def s1421_triton(a, b):
    n_elements = a.shape[0] // 2
    
    # xx points to &b[LEN_1D/2]
    xx = b[n_elements:]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1421_kernel[grid](
        a, b, xx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )