import torch
import triton
import triton.language as tl

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # This kernel must run with a single program to maintain sequential dependencies
    # Each iteration depends on the previous one through a[i+1] = ... and uses a[i]
    
    # Process all elements sequentially in a single thread
    if tl.program_id(0) == 0:
        for i in range(n_elements - 1):
            # Load values for current iteration
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            e_val = tl.load(e_ptr + i)
            a_val = tl.load(a_ptr + i)
            
            # Compute and store: a[i+1] = b[i] + c[i]
            a_next = b_val + c_val
            tl.store(a_ptr + i + 1, a_next)
            
            # Compute and store: b[i] = c[i] * e[i]
            b_new = c_val * e_val
            tl.store(b_ptr + i, b_new)
            
            # Compute and store: d[i] = a[i] * e[i]
            d_new = a_val * e_val
            tl.store(d_ptr + i, d_new)

def s3251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Launch with single program since computation must be sequential
    grid = (1,)
    
    s3251_kernel[grid](
        a, b, c, d, e,
        n_elements
    )