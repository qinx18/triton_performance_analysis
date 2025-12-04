import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(a_ptr, flat_2d_array_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Compute xx[i+1] = flat_2d_array[i] + a[i]
    result = flat_vals + a_vals
    
    # Store to xx[i+1] (offset by 1)
    store_idx = idx + 1
    store_mask = mask & (store_idx < (n_elements + 1))
    tl.store(xx_ptr + store_idx, result, mask=store_mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    vl = 63
    
    # Create xx array with same size as a (LEN_1D)
    xx = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s424_kernel[grid](
        a, flat_2d_array, xx, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return xx