import triton
import triton.language as tl
import torch

@triton.jit
def vtvtv_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    result = a_vals * b_vals * c_vals
    
    tl.store(a_ptr + idx, result, mask=mask)

def vtvtv_triton(a, b, c):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    vtvtv_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a