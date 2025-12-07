import triton
import triton.language as tl
import torch

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < m
    
    for j in range(m):
        c_val = tl.load(c_ptr + j)
        
        b_idx = i_idx + m - j - 1
        b_vals = tl.load(b_ptr + b_idx, mask=i_mask, other=0.0)
        
        a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
        
        result = a_vals + b_vals * c_val
        
        tl.store(a_ptr + i_idx, result, mask=i_mask)

def s176_triton(a, b, c, m):
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](
        a, b, c, m,
        BLOCK_SIZE=BLOCK_SIZE
    )