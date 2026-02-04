import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Prologue: handle i=1 case
    pid = tl.program_id(0)
    if pid == 0:
        a_1 = tl.load(a_copy_ptr + 1)
        b_0 = tl.load(b_ptr + 0)
        c_1 = tl.load(c_ptr + 1)
        result = b_0 + c_1
        tl.store(a_ptr + 1, result)
    
    # Main parallel loop: i from 1 to n-3 inclusive
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE + 1
    i_offsets = block_start + offsets
    
    mask = (i_offsets >= 1) & (i_offsets < n - 2)
    
    # Step 1: compute b[i] = a_copy[i+1] * d[i]
    a_vals = tl.load(a_copy_ptr + i_offsets + 1, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    b_vals = a_vals * d_vals
    tl.store(b_ptr + i_offsets, b_vals, mask=mask)
    
    # Step 2: compute a[i+1] = b[i] + c[i+1]
    c_vals = tl.load(c_ptr + i_offsets + 1, mask=mask)
    a_new_vals = b_vals + c_vals
    tl.store(a_ptr + i_offsets + 1, a_new_vals, mask=mask)
    
    # Epilogue: handle i=n-2 case
    if pid == 0:
        i_last = n - 2
        a_val = tl.load(a_copy_ptr + i_last + 1)
        d_val = tl.load(d_ptr + i_last)
        result = a_val * d_val
        tl.store(b_ptr + i_last, result)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    
    # Create read-only copy of array a to avoid WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n - 2, BLOCK_SIZE)
    
    s1213_kernel[grid_size,](
        a, b, c, d, a_copy, n, BLOCK_SIZE
    )