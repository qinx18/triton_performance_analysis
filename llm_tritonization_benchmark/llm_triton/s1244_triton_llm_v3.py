import torch
import triton
import triton.language as tl

@triton.jit
def s1244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load b and c values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
    
    # Store a values
    tl.store(a_ptr + offsets, a_vals, mask=mask)

@triton.jit
def s1244_d_kernel(a_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load a[i] and a[i+1]
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    a_next_vals = tl.load(a_ptr + offsets + 1, mask=mask)
    
    # Compute d[i] = a[i] + a[i+1]
    d_vals = a_vals + a_next_vals
    
    # Store d values
    tl.store(d_ptr + offsets, d_vals, mask=mask)

def s1244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # First kernel: compute a[i] values
    s1244_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE)
    
    # Second kernel: compute d[i] values using updated a values
    s1244_d_kernel[grid](a, d, n_elements, BLOCK_SIZE)