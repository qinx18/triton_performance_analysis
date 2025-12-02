import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
    
    # Load a[i+1] values - need to handle boundary
    a_next_offsets = current_offsets + 1
    a_next_mask = a_next_offsets < n_elements
    a_next_vals = tl.load(a_copy_ptr + a_next_offsets, mask=a_next_mask)
    
    # First computation: a[i] = b[i] + c[i] * d[i]
    a_temp = b_vals + c_vals * d_vals
    
    # Second computation: b[i] = a[i] + d[i] * e[i]
    b_new = a_temp + d_vals * e_vals
    
    # Third computation: a[i] = b[i] + a[i+1] * d[i]
    a_new = b_new + a_next_vals * d_vals
    
    # Store results
    tl.store(a_ptr + current_offsets, a_new, mask=mask)
    tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s243_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    BLOCK_SIZE = 256
    
    # Create read-only copy for WAR dependency
    a_copy = a.clone()
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, a_copy, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )