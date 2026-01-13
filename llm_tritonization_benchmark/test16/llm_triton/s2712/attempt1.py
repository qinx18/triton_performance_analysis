import torch
import triton
import triton.language as tl

@triton.jit
def s2712_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < N
    
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    
    condition = a_vals > b_vals
    update = b_vals * c_vals
    result = tl.where(condition, a_vals + update, a_vals)
    
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s2712_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s2712_kernel[grid](a, b, c, N, BLOCK_SIZE)