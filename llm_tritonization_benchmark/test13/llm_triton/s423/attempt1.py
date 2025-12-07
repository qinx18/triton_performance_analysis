import torch
import triton
import triton.language as tl

@triton.jit
def s423_kernel(
    flat_2d_array_ptr,
    a_ptr,
    n_elements,
    vl: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Compute indices for the loop: i ranges from 0 to n_elements-1
    i_offsets = block_start + offsets
    mask = i_offsets < n_elements
    
    # Load a[i] and xx[i] where xx = flat_2d_array + vl
    a_vals = tl.load(a_ptr + i_offsets, mask=mask)
    xx_vals = tl.load(flat_2d_array_ptr + vl + i_offsets, mask=mask)
    
    # Compute flat_2d_array[i+1] = xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1]
    tl.store(flat_2d_array_ptr + i_offsets + 1, result, mask=mask)

def s423_triton(a, flat_2d_array):
    LEN_1D = a.shape[0]
    vl = 64
    n_elements = LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array,
        a,
        n_elements,
        vl,
        BLOCK_SIZE,
    )
    
    return flat_2d_array