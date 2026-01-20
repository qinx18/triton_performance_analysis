import triton
import triton.language as tl
import torch

@triton.jit
def s423_kernel(flat_2d_array_ptr, a_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # xx is flat_2d_array + 64, so xx[i] is flat_2d_array[i + 64]
    xx_indices = indices + 64
    xx_mask = xx_indices < (n + 64)
    xx_vals = tl.load(flat_2d_array_ptr + xx_indices, mask=xx_mask, other=0.0)
    
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1]
    output_indices = indices + 1
    output_mask = mask
    
    tl.store(flat_2d_array_ptr + output_indices, result, mask=output_mask)

def s423_triton(a, flat_2d_array):
    n = a.shape[0] - 1
    
    flat_2d_array_copy = flat_2d_array.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array_copy,
        a,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return flat_2d_array_copy