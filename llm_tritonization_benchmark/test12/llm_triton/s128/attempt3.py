import triton
import triton.language as tl
import torch

@triton.jit
def s128_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load d values
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Calculate k = 2 * i (where i is the offset)
    k_offsets = 2 * offsets
    k_mask = k_offsets < (2 * n_elements)
    
    # Load b and c values
    b_vals = tl.load(b_ptr + k_offsets, mask=k_mask)
    c_vals = tl.load(c_ptr + k_offsets, mask=k_mask)
    
    # Compute a[i] = b[k] - d[i]
    a_vals = b_vals - d_vals
    
    # Store a values
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Compute b[k] = a[i] + c[k]
    b_new_vals = a_vals + c_vals
    
    # Store updated b values
    tl.store(b_ptr + k_offsets, b_new_vals, mask=k_mask)

def s128_triton(a, b, c, d):
    n_elements = a.shape[0] // 2
    BLOCK_SIZE = 64
    
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s128_kernel[(grid_size,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b