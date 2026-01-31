import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # Use single thread for sequential processing
    thread_id = tl.program_id(0)
    
    # Only first thread does the work
    if thread_id != 0:
        return
    
    # Sequential loop - cannot be parallelized due to loop-carried dependency
    for i in range(1, n_elements - 1):
        # Load values for current iteration
        b_prev = tl.load(b_ptr + (i - 1))
        b_next = tl.load(b_ptr + (i + 1))
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_val = b_prev + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Compute b[i] = b[i+1] - e[i] * d[i]
        b_val = b_next - e_val * d_val
        tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Launch with single thread since this must be sequential
    grid = (1,)
    
    s211_kernel[grid](
        a, b, c, d, e,
        N
    )