import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load arrays
    i_offsets = block_start + offsets
    k_offsets = 2 * i_offsets + 1
    
    # Check bounds for k indices
    k_mask = mask & (k_offsets < 2 * n_elements)
    
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + k_offsets, mask=k_mask, other=0.0)
    c_vals = tl.load(c_ptr + k_offsets, mask=k_mask, other=0.0)
    
    # Compute: a[i] = b[k] - d[i] where k = 2*i + 1
    a_vals = b_vals - d_vals
    
    # Store a[i]
    tl.store(a_ptr + i_offsets, a_vals, mask=mask)
    
    # Compute: b[k] = a[i] + c[k]
    b_new_vals = a_vals + c_vals
    
    # Store b[k]
    tl.store(b_ptr + k_offsets, b_new_vals, mask=k_mask)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s128_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )