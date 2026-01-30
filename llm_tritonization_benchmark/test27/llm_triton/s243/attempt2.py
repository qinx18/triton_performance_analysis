import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load all required values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Load a[i+1] values from read-only copy
    next_indices = indices + 1
    next_mask = next_indices < (n_elements + 1)
    a_next_vals = tl.load(a_copy_ptr + next_indices, mask=next_mask)
    
    # First operation: a[i] = b[i] + c[i] * d[i]
    a_temp = b_vals + c_vals * d_vals
    
    # Second operation: b[i] = a[i] + d[i] * e[i]
    b_temp = a_temp + d_vals * e_vals
    tl.store(b_ptr + indices, b_temp, mask=mask)
    
    # Third operation: a[i] = b[i] + a[i+1] * d[i]
    a_final = b_temp + a_next_vals * d_vals
    tl.store(a_ptr + indices, a_final, mask=mask)

def s243_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, b, c, d, e, a_copy,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE
    )