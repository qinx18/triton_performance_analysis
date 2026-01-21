import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Load a[i+1] values from copy for third statement
    a_plus1_indices = indices + 1
    a_plus1_mask = mask & (a_plus1_indices < (n + 1))
    a_plus1_vals = tl.load(a_copy_ptr + a_plus1_indices, mask=a_plus1_mask, other=0.0)
    
    # First statement: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # Second statement: b[i] = a[i] + d[i] * e[i]
    b_vals = a_vals + d_vals * e_vals
    tl.store(b_ptr + indices, b_vals, mask=mask)
    
    # Third statement: a[i] = b[i] + a[i+1] * d[i]
    a_vals = b_vals + a_plus1_vals * d_vals
    tl.store(a_ptr + indices, a_vals, mask=mask)

def s243_triton(a, b, c, d, e):
    n = a.shape[0] - 1
    
    # Create read-only copy for WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, a_copy, b, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )