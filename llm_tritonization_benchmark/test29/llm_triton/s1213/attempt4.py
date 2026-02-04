import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Handle range [1, n-2] for the main transformed loop
    mask = (offsets >= 1) & (offsets < n-1)
    
    # Producer: b[i] = a[i+1] * d[i]
    a_next = tl.load(a_copy_ptr + offsets + 1, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_vals = a_next * d_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    
    # Consumer: a[i+1] = b[i] + c[i+1]
    mask_consumer = (offsets >= 1) & (offsets < n-2)
    c_next = tl.load(c_ptr + offsets + 1, mask=mask_consumer)
    a_new = b_vals + c_next
    tl.store(a_ptr + offsets + 1, a_new, mask=mask_consumer)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    a_copy = a.clone()
    
    # Prologue: a[1] = b[0] + c[1]
    if n > 1:
        a[1] = b[0] + c[1]
    
    # Main parallel loop
    if n > 2:
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        s1213_kernel[grid](a, a_copy, b, c, d, n, BLOCK_SIZE)