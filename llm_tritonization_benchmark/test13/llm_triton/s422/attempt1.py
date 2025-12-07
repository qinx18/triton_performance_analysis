import triton
import triton.language as tl
import torch

@triton.jit
def s422_kernel(
    a_ptr,
    flat_2d_array_ptr,
    xx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load a[i]
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Load flat_2d_array[i + 8]
    flat_idx = idx + 8
    flat_mask = mask & (flat_idx < (n_elements + 8))
    flat_vals = tl.load(flat_2d_array_ptr + flat_idx, mask=flat_mask)
    
    # Compute xx[i] = flat_2d_array[i + 8] + a[i]
    result = flat_vals + a_vals
    
    # Store to xx[i]
    tl.store(xx_ptr + idx, result, mask=mask)

def s422_triton(a, flat_2d_array, xx):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s422_kernel[grid](
        a,
        flat_2d_array,
        xx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return xx