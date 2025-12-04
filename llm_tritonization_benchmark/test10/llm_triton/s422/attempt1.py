import triton
import triton.language as tl
import torch

@triton.jit
def s422_kernel(a_ptr, flat_2d_array_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load a[i] and flat_2d_array[i + 8]
    a_vals = tl.load(a_ptr + idx, mask=mask)
    flat_vals = tl.load(flat_2d_array_ptr + idx + 8, mask=mask)
    
    # Compute xx[i] = flat_2d_array[i + 8] + a[i]
    result = flat_vals + a_vals
    
    # Store to xx[i]
    tl.store(xx_ptr + idx, result, mask=mask)

def s422_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s422_kernel[grid](
        a, flat_2d_array, xx, n_elements, BLOCK_SIZE
    )