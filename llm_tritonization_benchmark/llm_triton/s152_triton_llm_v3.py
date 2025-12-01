import torch
import triton
import triton.language as tl

@triton.jit
def s152s_kernel(a_ptr, b_ptr, c_ptr, idx, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values from arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Get b[idx] and c[idx] values
    b_idx_val = tl.load(b_ptr + idx)
    c_idx_val = tl.load(c_ptr + idx)
    
    # Compute: a[j] = (a[j] + b[idx]) * c[idx] for all j
    result = (a_vals + b_idx_val) * c_idx_val
    
    # Store back to a
    tl.store(a_ptr + offsets, result, mask=mask)

@triton.jit
def s152_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # b[i] = d[i] * e[i]
    d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
    b_vals = d_vals * e_vals
    tl.store(b_ptr + current_offsets, b_vals, mask=mask)

def s152_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # First pass: compute b[i] = d[i] * e[i] for all i
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s152_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE
    )
    
    # Second pass: call s152s for each i
    for i in range(n_elements):
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        s152s_kernel[grid](
            a, b, c, i, n_elements, BLOCK_SIZE
        )