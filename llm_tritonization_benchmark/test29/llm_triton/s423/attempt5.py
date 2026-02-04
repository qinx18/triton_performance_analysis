import triton
import triton.language as tl
import torch

@triton.jit
def s423_kernel(flat_2d_array_ptr, xx_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    xx_vals = tl.load(xx_ptr + offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    result = xx_vals + a_vals
    
    output_offsets = offsets + 1
    output_mask = output_offsets < (n_elements + 1)
    tl.store(flat_2d_array_ptr + output_offsets, result, mask=output_mask)

def s423_triton(a, flat_2d_array, xx):
    N = a.shape[0]
    n_elements = N - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array,
        xx,
        a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )