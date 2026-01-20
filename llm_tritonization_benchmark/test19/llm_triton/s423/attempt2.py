import torch
import triton
import triton.language as tl

@triton.jit
def s423_kernel(flat_2d_array_ptr, a_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load a[i] and xx[i] (where xx = flat_2d_array + 64)
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    xx_vals = tl.load(flat_2d_array_ptr + 64 + indices, mask=mask, other=0.0)
    
    # Compute flat_2d_array[i+1] = xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1]
    tl.store(flat_2d_array_ptr + indices + 1, result, mask=mask)

def s423_triton(a, flat_2d_array, xx):
    n = a.shape[0] - 1  # Loop goes from 0 to LEN_1D - 2 (i < LEN_1D - 1)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array, a, n, 
        BLOCK_SIZE=BLOCK_SIZE
    )