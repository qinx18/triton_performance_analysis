import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(
    a_ptr,
    flat_2d_array_ptr,
    xx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    # Compute xx[i+1] = flat_2d_array[i] + a[i]
    result = flat_vals + a_vals
    
    # Store to xx[i+1]
    store_indices = indices + 1
    store_mask = mask & (store_indices < n_elements + 1)
    tl.store(xx_ptr + store_indices, result, mask=store_mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    # Create xx array (offset version of flat_2d_array)
    vl = 63
    xx = torch.zeros_like(flat_2d_array)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s424_kernel[grid](
        a,
        flat_2d_array,
        xx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return xx