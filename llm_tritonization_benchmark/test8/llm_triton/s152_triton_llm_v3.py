import torch
import triton
import triton.language as tl

@triton.jit
def s152s_kernel(a_ptr, b_ptr, c_ptr, idx, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Get b[idx] value
    b_idx_offset = tl.arange(0, 1) + idx
    b_idx_mask = b_idx_offset < n_elements
    b_idx_val = tl.load(b_ptr + b_idx_offset, mask=b_idx_mask)
    
    # Update a based on the pattern: a[j] += b[idx] * c[j]
    a_vals = a_vals + b_idx_val * c_vals
    
    # Store back
    tl.store(a_ptr + offsets, a_vals, mask=mask)

@triton.jit
def s152_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # b[i] = d[i] * e[i]
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    b_vals = d_vals * e_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s152_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # First compute b[i] = d[i] * e[i] for all i
    s152_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE)
    
    # Then call s152s for each i (this must be sequential due to data dependencies)
    for i in range(n_elements):
        s152s_grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        s152s_kernel[s152s_grid](a, b, c, i, n_elements, BLOCK_SIZE)