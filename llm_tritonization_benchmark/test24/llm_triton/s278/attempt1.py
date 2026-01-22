import triton
import triton.language as tl
import torch

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load all arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Create condition mask: a[i] > 0
    condition = a_vals > 0.0
    
    # Compute both branches
    # If condition false: b[i] = -b[i] + d[i] * e[i]
    b_new_false = -b_vals + d_vals * e_vals
    # If condition true: c[i] = -c[i] + d[i] * e[i]
    c_new_true = -c_vals + d_vals * e_vals
    
    # Select values based on condition
    b_new = tl.where(condition, b_vals, b_new_false)
    c_new = tl.where(condition, c_new_true, c_vals)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_new = b_new + c_new * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)
    tl.store(c_ptr + offsets, c_new, mask=mask)

def s278_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s278_kernel[grid](a, b, c, d, e, N, BLOCK_SIZE=BLOCK_SIZE)