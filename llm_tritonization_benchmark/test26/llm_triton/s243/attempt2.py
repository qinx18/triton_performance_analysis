import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values for current iteration
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
    
    # Load a[i+1] from the read-only copy
    next_offsets = current_offsets + 1
    next_mask = next_offsets < (n_elements + 1)
    a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask, other=0.0)
    
    # First statement: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    
    # Second statement: b[i] = a[i] + d[i] * e[i]
    b_vals_new = a_vals + d_vals * e_vals
    
    # Third statement: a[i] = b[i] + a[i+1] * d[i]
    a_vals_final = b_vals_new + a_next_vals * d_vals
    
    # Store final results
    tl.store(a_ptr + current_offsets, a_vals_final, mask=mask)
    tl.store(b_ptr + current_offsets, b_vals_new, mask=mask)

def s243_triton(a, b, c, d, e):
    n = a.shape[0]
    n_elements = n - 1  # Loop runs from 0 to LEN_1D-2
    
    # Create read-only copy of 'a' to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, b, c, d, e, a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )