import triton
import triton.language as tl
import torch

@triton.jit
def s423_kernel(flat_2d_array_ptr, xx_ptr, a_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    xx_vals = tl.load(xx_ptr + indices, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    result = xx_vals + a_vals
    
    output_indices = indices + 1
    output_mask = output_indices < (n + 1)
    
    tl.store(flat_2d_array_ptr + output_indices, result, mask=output_mask)

def s423_triton(a, flat_2d_array, xx):
    n = a.shape[0] - 1
    
    flat_2d_array_copy = flat_2d_array.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array_copy,
        xx,
        a,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return flat_2d_array_copy