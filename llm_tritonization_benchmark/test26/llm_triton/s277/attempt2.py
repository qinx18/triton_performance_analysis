import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # Process sequentially due to loop-carried dependency
    for i in range(n_elements - 1):
        # Check if a[i] >= 0
        a_val = tl.load(a_ptr + i)
        skip_all = a_val >= 0.0
        
        # Check if b[i] >= 0 (only if not skipping all)
        b_val = tl.load(b_ptr + i)
        skip_a_update = (b_val >= 0.0) | skip_all
        
        # Execute a[i] += c[i] * d[i] (only if not skipping)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        a_new = tl.where(skip_a_update, a_val, a_val + c_val * d_val)
        tl.store(a_ptr + i, a_new)
        
        # Execute L30: b[i+1] = c[i] + d[i] * e[i] (only if not skipping all)
        e_val = tl.load(e_ptr + i)
        b_new = c_val + d_val * e_val
        b_old = tl.load(b_ptr + i + 1)
        b_final = tl.where(skip_all, b_old, b_new)
        tl.store(b_ptr + i + 1, b_final)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Launch single thread to handle sequential dependency
    grid = (1,)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements
    )