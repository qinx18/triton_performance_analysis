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
    
    # Load data
    xx_vals = tl.load(xx_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Compute
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1]
    store_indices = indices + 1
    store_mask = mask
    tl.store(flat_2d_array_ptr + store_indices, result, mask=store_mask)

def s423_triton(a, flat_2d_array, xx):
    N = a.shape[0] - 1  # Loop goes to LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array, xx, a, N, 
        BLOCK_SIZE=BLOCK_SIZE
    )