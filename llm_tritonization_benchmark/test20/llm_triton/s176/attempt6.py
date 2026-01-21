import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < m
    
    for j in range(m):
        b_indices = i_idx + m - j - 1
        b_mask = i_mask & (b_indices >= 0) & (b_indices < 2 * m)
        
        a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
        b_vals = tl.load(b_ptr + b_indices, mask=b_mask, other=0.0)
        c_val = tl.load(c_ptr + j)
        
        result = a_vals + b_vals * c_val
        tl.store(a_ptr + i_idx, result, mask=i_mask)

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](a, b, c, m, BLOCK_SIZE=BLOCK_SIZE)