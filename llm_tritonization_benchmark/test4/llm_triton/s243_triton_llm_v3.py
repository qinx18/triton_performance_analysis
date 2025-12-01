import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(axis=0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Load a[i+1] from copy (need to handle boundary)
    a_next_mask = (idx + 1) < n_elements
    a_next_vals = tl.load(a_copy_ptr + idx + 1, mask=a_next_mask)
    
    # First statement: a[i] = b[i] + c[i] * d[i]
    a_temp1 = b_vals + c_vals * d_vals
    
    # Second statement: b[i] = a[i] + d[i] * e[i]
    b_new = a_temp1 + d_vals * e_vals
    
    # Third statement: a[i] = b[i] + a[i+1] * d[i]
    a_final = b_new + a_next_vals * d_vals
    
    # Store results
    tl.store(a_ptr + idx, a_final, mask=mask)
    tl.store(b_ptr + idx, b_new, mask=mask)

def s243_triton(a, b, c, d, e):
    n_elements = len(a) - 1  # Process LEN_1D-1 elements
    
    # Create read-only copy of array a for WAR race condition handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, a_copy, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )