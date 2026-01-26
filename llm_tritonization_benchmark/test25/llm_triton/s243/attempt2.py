import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Load a[i+1] from copy
    next_indices = indices + 1
    next_mask = next_indices < (n_elements + 1)
    a_next_vals = tl.load(a_copy_ptr + next_indices, mask=next_mask)
    
    # First statement: a[i] = b[i] + c[i] * d[i]
    a_new = b_vals + c_vals * d_vals
    
    # Second statement: b[i] = a[i] + d[i] * e[i]
    b_new = a_new + d_vals * e_vals
    
    # Third statement: a[i] = b[i] + a[i+1] * d[i]
    a_final = b_new + a_next_vals * d_vals
    
    # Store results
    tl.store(a_ptr + indices, a_final, mask=mask)
    tl.store(b_ptr + indices, b_new, mask=mask)

def s243_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy of array a
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, a_copy, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )