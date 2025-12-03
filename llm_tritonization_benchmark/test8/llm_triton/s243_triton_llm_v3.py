import torch
import triton
import triton.language as tl

@triton.jit
def s243_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, a_copy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    
    # Read a[i+1] from copy (need to handle boundary)
    next_offsets = current_offsets + 1
    next_mask = next_offsets < (n_elements + 1)  # Allow reading a[i+1]
    a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask, other=0.0)
    
    # First assignment: a[i] = b[i] + c[i] * d[i]
    a_new = b_vals + c_vals * d_vals
    tl.store(a_ptr + current_offsets, a_new, mask=mask)
    
    # Second assignment: b[i] = a[i] + d[i] * e[i]
    b_new = a_new + d_vals * e_vals
    tl.store(b_ptr + current_offsets, b_new, mask=mask)
    
    # Third assignment: a[i] = b[i] + a[i+1] * d[i]
    a_final = b_new + a_next_vals * d_vals
    tl.store(a_ptr + current_offsets, a_final, mask=mask)

def s243_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy of a for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, b, c, d, e, a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )