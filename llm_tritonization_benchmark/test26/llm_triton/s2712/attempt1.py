import triton
import triton.language as tl
import torch

@triton.jit
def s2712_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        condition = a_vals > b_vals
        update_vals = b_vals * c_vals
        new_a_vals = tl.where(condition, a_vals + update_vals, a_vals)
        
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)

def s2712_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    s2712_kernel[(1,)](a, b, c, N, BLOCK_SIZE)