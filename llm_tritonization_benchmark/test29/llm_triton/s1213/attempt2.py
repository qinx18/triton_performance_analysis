import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = (offsets >= 1) & (offsets < n-1)
    
    # Statement S0: a[i] = b[i-1] + c[i]
    b_prev = tl.load(b_ptr + offsets - 1, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = b_prev + c_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Statement S1: b[i] = a[i+1] * d[i]
    a_next = tl.load(a_copy_ptr + offsets + 1, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_vals = a_next * d_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1213_kernel[grid](
        a, a_copy, b, c, d, n, BLOCK_SIZE
    )