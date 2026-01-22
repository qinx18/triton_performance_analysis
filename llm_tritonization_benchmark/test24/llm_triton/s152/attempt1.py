import triton
import triton.language as tl
import torch

@triton.jit
def s152_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < N
    
    # b[i] = d[i] * e[i]
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    b_vals = d_vals * e_vals
    tl.store(b_ptr + indices, b_vals, mask=mask)
    
    # a[i] += b[i] * c[i] (inlined s152s function)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    result = a_vals + b_vals * c_vals
    tl.store(a_ptr + indices, result, mask=mask)

def s152_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s152_kernel[grid](a, b, c, d, e, N, BLOCK_SIZE=BLOCK_SIZE)