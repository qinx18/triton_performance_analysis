import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    i = offsets
    k = 2 * i
    
    # Check bounds for k indices
    k_mask = k < (2 * n_elements)
    combined_mask = mask & k_mask
    
    # Load data
    d_vals = tl.load(d_ptr + i, mask=mask)
    b_vals = tl.load(b_ptr + k, mask=combined_mask)
    c_vals = tl.load(c_ptr + k, mask=combined_mask)
    
    # Compute a[i] = b[k] - d[i]
    a_vals = b_vals - d_vals
    
    # Store a[i]
    tl.store(a_ptr + i, a_vals, mask=mask)
    
    # Compute b[k] = a[i] + c[k]
    b_new_vals = a_vals + c_vals
    
    # Store b[k]
    tl.store(b_ptr + k, b_new_vals, mask=combined_mask)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b