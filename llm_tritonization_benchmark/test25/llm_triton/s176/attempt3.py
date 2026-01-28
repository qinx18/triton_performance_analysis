import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask_i = i_idx < m
    
    a_vals = tl.load(a_ptr + i_idx, mask=mask_i, other=0.0)
    
    for j in range(m):
        b_idx = i_idx + m - j - 1
        b_vals = tl.load(b_ptr + b_idx, mask=mask_i, other=0.0)
        c_val = tl.load(c_ptr + j)
        a_vals += b_vals * c_val
    
    tl.store(a_ptr + i_idx, a_vals, mask=mask_i)

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](a, b, c, m, BLOCK_SIZE=BLOCK_SIZE)