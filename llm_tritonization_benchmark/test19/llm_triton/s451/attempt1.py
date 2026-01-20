import triton
import triton.language as tl
import torch

@triton.jit
def s451_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    sin_b = tl.sin(b_vals)
    cos_c = tl.cos(c_vals)
    result = sin_b + cos_c
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s451_triton(a, b, c, cosf, sinf):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s451_kernel[grid](a, b, c, n, BLOCK_SIZE=BLOCK_SIZE)