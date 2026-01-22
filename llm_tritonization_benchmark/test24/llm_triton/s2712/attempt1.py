import triton
import triton.language as tl
import torch

@triton.jit
def s2712_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    condition = a_vals > b_vals
    update = b_vals * c_vals
    new_a_vals = tl.where(condition, a_vals + update, a_vals)
    
    tl.store(a_ptr + offsets, new_a_vals, mask=mask)

def s2712_triton(a, b, c):
    n = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s2712_kernel[grid](a, b, c, n, BLOCK_SIZE=BLOCK_SIZE)