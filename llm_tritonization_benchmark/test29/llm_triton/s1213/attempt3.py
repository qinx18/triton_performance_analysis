import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel_s0(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets >= 1) & (offsets < n-1)
    
    b_prev = tl.load(b_ptr + offsets - 1, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = b_prev + c_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)

@triton.jit
def s1213_kernel_s1(b_ptr, a_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets >= 1) & (offsets < n-1)
    
    a_next = tl.load(a_ptr + offsets + 1, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_vals = a_next * d_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # First loop: a[i] = b[i-1] + c[i]
    s1213_kernel_s0[grid](a, b, c, n, BLOCK_SIZE)
    
    # Second loop: b[i] = a[i+1] * d[i]
    s1213_kernel_s1[grid](b, a, d, n, BLOCK_SIZE)