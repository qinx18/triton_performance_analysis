import torch
import triton
import triton.language as tl

@triton.jit
def s423_kernel(a_ptr, flat_2d_array_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Current indices for this block
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Read from a[i] and xx[i] where xx = flat_2d_array + 64
    a_vals = tl.load(a_ptr + indices, mask=mask)
    xx_vals = tl.load(flat_2d_array_ptr + 64 + indices, mask=mask)
    
    # Compute flat_2d_array[i+1] = xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Write to flat_2d_array[i+1]
    write_indices = indices + 1
    write_mask = write_indices < (n_elements + 1)
    tl.store(flat_2d_array_ptr + write_indices, result, mask=write_mask)

def s423_triton(a, flat_2d_array):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        a, flat_2d_array, n_elements, BLOCK_SIZE
    )