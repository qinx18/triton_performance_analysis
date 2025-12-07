import torch
import triton
import triton.language as tl

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate current indices
    i_offsets = block_start + offsets
    mask_i = i_offsets < n_elements
    
    # Calculate k indices (k = 2*i for each i)
    k_offsets = 2 * i_offsets
    mask_k = k_offsets < (2 * n_elements)
    
    # Load d[i] and c[k]
    d_vals = tl.load(d_ptr + i_offsets, mask=mask_i)
    c_vals = tl.load(c_ptr + k_offsets, mask=mask_k)
    
    # Load b[k] 
    b_vals = tl.load(b_ptr + k_offsets, mask=mask_k)
    
    # Compute a[i] = b[k] - d[i]
    a_vals = b_vals - d_vals
    
    # Store a[i]
    tl.store(a_ptr + i_offsets, a_vals, mask=mask_i)
    
    # Compute b[k] = a[i] + c[k]
    b_new_vals = a_vals + c_vals
    
    # Store b[k]
    tl.store(b_ptr + k_offsets, b_new_vals, mask=mask_k)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )