import triton
import triton.language as tl
import torch

@triton.jit
def s174_kernel(a_ptr, b_ptr, M, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, M, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < M
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        
        result = a_vals + b_vals
        
        tl.store(a_ptr + (current_offsets + M), result, mask=mask)

def s174_triton(a, b):
    M = a.shape[0] // 2
    BLOCK_SIZE = 256
    
    s174_kernel[(1,)](a, b, M, BLOCK_SIZE=BLOCK_SIZE)
    
    return a