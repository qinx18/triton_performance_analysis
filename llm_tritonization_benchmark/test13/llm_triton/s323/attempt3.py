import torch
import triton
import triton.language as tl

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # Get the current strip (each strip processes 1 element)
    strip_id = tl.program_id(0)
    
    # Calculate the actual index (starting from 1)
    idx = strip_id + 1
    
    if idx < n_elements:
        # Load values
        b_prev = tl.load(b_ptr + idx - 1)
        c_val = tl.load(c_ptr + idx)
        d_val = tl.load(d_ptr + idx)
        e_val = tl.load(e_ptr + idx)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_val = b_prev + c_val * d_val
        
        # Compute b[i] = a[i] + c[i] * e[i]
        b_val = a_val + c_val * e_val
        
        # Store results
        tl.store(a_ptr + idx, a_val)
        tl.store(b_ptr + idx, b_val)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Number of elements to process (from 1 to n_elements-1)
    num_strips = n_elements - 1
    
    if num_strips > 0:
        # Launch kernel with one thread per strip
        s323_kernel[(num_strips,)](
            a, b, c, d, e,
            n_elements
        )