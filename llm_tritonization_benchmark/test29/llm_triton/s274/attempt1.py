import triton
import triton.language as tl
import torch

@triton.jit
def s274_kernel_1(a_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    a_vals = c_vals + e_vals * d_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)

@triton.jit
def s274_kernel_2(a_ptr, b_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Condition: a[i] > 0
    cond = a_vals > 0.0
    
    # If true: b[i] = a[i] + b[i] 
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    b_new = a_vals + b_vals
    
    # If false: a[i] = d[i] * e[i]
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_new = d_vals * e_vals
    
    # Store conditionally
    tl.store(b_ptr + offsets, b_new, mask=mask & cond)
    tl.store(a_ptr + offsets, a_new, mask=mask & (~cond))

def s274_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # First kernel: a[i] = c[i] + e[i] * d[i]
    s274_kernel_1[grid](a, c, d, e, N, BLOCK_SIZE)
    
    # Second kernel: conditional update
    s274_kernel_2[grid](a, b, d, e, N, BLOCK_SIZE)