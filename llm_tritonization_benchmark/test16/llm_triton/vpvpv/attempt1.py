import torch
import triton
import triton.language as tl

@triton.jit
def vpvpv_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    result = a_vals + b_vals + c_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

def vpvpv_triton(a, b, c):
    N = a.shape[0]
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    vpvpv_kernel[grid](
        a, b, c,
        N,
        BLOCK_SIZE=BLOCK_SIZE
    )