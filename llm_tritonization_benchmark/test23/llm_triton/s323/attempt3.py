import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # This is a strictly sequential loop due to the dependency b[i] depends on b[i-1]
    for i in range(1, n_elements):
        # Load required values
        b_prev = tl.load(b_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_val = b_prev + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Compute b[i] = a[i] + c[i] * e[i]
        b_val = a_val + c_val * e_val
        tl.store(b_ptr + i, b_val)

def s323_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Launch single thread to handle sequential computation
    grid = (1,)
    
    s323_kernel[grid](
        a, b, c, d, e, N
    )