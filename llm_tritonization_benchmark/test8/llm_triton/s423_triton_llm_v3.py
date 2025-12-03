import triton
import triton.language as tl
import torch

@triton.jit
def s423_kernel(flat_2d_array_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load from a[i]
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Load from xx[i] (which is flat_2d_array + 64)
    xx_vals = tl.load(flat_2d_array_ptr + 64 + indices, mask=mask)
    
    # Compute xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1]
    tl.store(flat_2d_array_ptr + indices + 1, result, mask=mask)

def s423_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](flat_2d_array, a, n_elements, BLOCK_SIZE)