import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    flat_vals = tl.load(flat_2d_array_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    result = flat_vals + a_vals
    
    output_indices = indices + 1
    output_mask = output_indices < (n_elements + 1)
    tl.store(xx_ptr + output_indices, result, mask=output_mask & mask)

def s424_triton(flat_2d_array, a, xx):
    n_elements = flat_2d_array.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s424_kernel[grid](
        flat_2d_array,
        a, 
        xx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return xx