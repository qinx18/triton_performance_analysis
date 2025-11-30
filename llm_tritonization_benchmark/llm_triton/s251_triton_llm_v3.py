import triton
import triton.language as tl
import torch

@triton.jit
def s251_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    
    s = b + c * d
    result = s * s
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s251_triton(a, b, c, d):
    n_elements = a.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    s251_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=1024,
    )
    
    return a