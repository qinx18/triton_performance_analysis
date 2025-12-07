import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, i: tl.constexpr):
    # Process single element at index i
    b_val = tl.load(b_ptr + i)
    
    if b_val < 0.0:
        # Path: c[i+1] = a[i] + d[i] * d[i]
        a_val = tl.load(a_ptr + i)
        d_val = tl.load(d_ptr + i)
        result = a_val + d_val * d_val
        tl.store(c_ptr + i + 1, result)
    else:
        # Path: a[i] = c[i] + d[i] * e[i]
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        result = c_val + d_val * e_val
        tl.store(a_ptr + i, result)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1
    
    # Process each element sequentially due to RAW dependency
    for i in range(n_elements):
        grid = (1,)
        s161_kernel[grid](a, b, c, d, e, i)