import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # This kernel must run sequentially due to RAW dependency in b[i+1] = ... pattern
    # Use single thread processing
    
    for i in range(n_elements - 1):
        # Load current values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        
        # Check first condition: if a[i] >= 0, skip to L20 (end)
        skip_to_l20 = a_val >= 0.0
        
        # Check second condition: if b[i] >= 0, skip to L30
        skip_to_l30 = b_val >= 0.0
        
        # Load values needed for calculations
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # If both conditions false, execute: a[i] += c[i] * d[i]
        should_update_a = (skip_to_l20 == 0) & (skip_to_l30 == 0)
        if should_update_a:
            a_val = a_val + c_val * d_val
            tl.store(a_ptr + i, a_val)
        
        # L30: b[i+1] = c[i] + d[i] * e[i] (execute unless skip_to_l20)
        if skip_to_l20 == 0:
            e_val = tl.load(e_ptr + i)
            result = c_val + d_val * e_val
            tl.store(b_ptr + i + 1, result)
        
        # L20: (empty label)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Launch single thread kernel due to sequential dependency
    grid = (1,)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements
    )