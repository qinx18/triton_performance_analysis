import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements):
    # This kernel must run with a single thread due to strict sequential dependency
    # b[i] = b[i-1] + a[i] + d[i] creates a loop-carried dependency
    
    for i in range(1, n_elements):
        # Load values for current iteration
        a_val = tl.load(a_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        b_prev = tl.load(b_ptr + i - 1)
        
        # Compute a[i] += c[i] * d[i]
        new_a_val = a_val + c_val * d_val
        tl.store(a_ptr + i, new_a_val)
        
        # Compute b[i] = b[i-1] + a[i] + d[i]
        new_b_val = b_prev + new_a_val + d_val
        tl.store(b_ptr + i, new_b_val)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Launch with single thread due to sequential dependency
    grid = (1,)
    s221_kernel[grid](
        a, b, c, d,
        n_elements
    )