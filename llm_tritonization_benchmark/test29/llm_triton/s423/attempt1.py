import triton
import triton.language as tl
import torch

@triton.jit
def s423_kernel(flat_2d_array_ptr, xx_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    i_offsets = block_start + offsets
    mask = i_offsets < n_elements
    
    xx_vals = tl.load(xx_ptr + i_offsets, mask=mask)
    a_vals = tl.load(a_ptr + i_offsets, mask=mask)
    
    result = xx_vals + a_vals
    
    output_offsets = i_offsets + 1
    output_mask = mask
    tl.store(flat_2d_array_ptr + output_offsets, result, mask=output_mask)

def s423_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array, xx, a, n_elements, BLOCK_SIZE
    )