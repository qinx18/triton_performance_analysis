import torch
import triton
import triton.language as tl

@triton.jit
def s423_kernel(
    flat_2d_array_ptr,
    xx_ptr,
    a_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load xx[i] and a[i]
    xx_vals = tl.load(xx_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Compute xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1]
    output_indices = indices + 1
    output_mask = output_indices < (n_elements + 1)
    tl.store(flat_2d_array_ptr + output_indices, result, mask=mask & output_mask)

def s423_triton(a, flat_2d_array, xx):
    N = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array,
        xx,
        a,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )