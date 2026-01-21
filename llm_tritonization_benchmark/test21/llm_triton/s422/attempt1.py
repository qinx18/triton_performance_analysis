import triton
import triton.language as tl
import torch

@triton.jit
def s422_kernel(a_ptr, flat_2d_array_ptr, xx_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < N
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    flat_2d_array_vals = tl.load(flat_2d_array_ptr + indices + 8, mask=mask)
    
    result = flat_2d_array_vals + a_vals
    
    tl.store(xx_ptr + indices, result, mask=mask)

def s422_triton(a, flat_2d_array, xx):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s422_kernel[grid](
        a, flat_2d_array, xx,
        N, BLOCK_SIZE
    )