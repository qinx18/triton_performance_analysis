import torch
import triton
import triton.language as tl

@triton.jit
def s423_kernel(flat_2d_array_ptr, xx_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    xx_vals = tl.load(xx_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    result = xx_vals + a_vals
    
    tl.store(flat_2d_array_ptr + (indices + 1), result, mask=mask)

def s423_triton(flat_2d_array, a):
    n_elements = len(a) - 1
    vl = 64
    xx = flat_2d_array[vl:]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array, xx, a, n_elements, BLOCK_SIZE
    )