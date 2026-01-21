import triton
import triton.language as tl
import torch

@triton.jit
def s422_kernel(a_ptr, flat_2d_array_ptr, xx_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    flat_vals = tl.load(flat_2d_array_ptr + (offsets + 8), mask=mask)
    
    # Compute
    result = flat_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + offsets, result, mask=mask)

def s422_triton(a, flat_2d_array, xx):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s422_kernel[grid](a, flat_2d_array, xx, N, BLOCK_SIZE=BLOCK_SIZE)