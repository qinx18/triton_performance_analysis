import triton
import triton.language as tl
import torch

@triton.jit
def s423_kernel(flat_2d_array_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load a[i]
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Load xx[i] which is flat_2d_array[i + 64]
    xx_indices = indices + 64
    xx_mask = xx_indices < (n_elements + 64)
    xx_vals = tl.load(flat_2d_array_ptr + xx_indices, mask=xx_mask, other=0.0)
    
    # Compute xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1]
    store_indices = indices + 1
    store_mask = store_indices < (n_elements + 1)
    tl.store(flat_2d_array_ptr + store_indices, result, mask=mask & store_mask)

def s423_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array, a, n_elements, BLOCK_SIZE
    )
    
    return flat_2d_array