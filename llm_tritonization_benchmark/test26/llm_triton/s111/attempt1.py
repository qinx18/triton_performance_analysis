import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements):
    # This kernel must process sequentially due to loop-carried dependency
    # Use single thread to maintain dependency chain
    thread_id = tl.program_id(0)
    
    # Only first thread processes to maintain sequential dependency
    if thread_id == 0:
        # Process sequentially: a[i] = a[i-1] + b[i] for i = 1, 3, 5, ...
        i = 1
        while i < n_elements:
            # Load a[i-1] and b[i]
            a_prev = tl.load(a_ptr + (i - 1))
            b_curr = tl.load(b_ptr + i)
            
            # Compute and store a[i]
            result = a_prev + b_curr
            tl.store(a_ptr + i, result)
            
            i += 2

def s111_triton(a, b):
    N = a.shape[0]
    
    # Launch with single thread to maintain sequential dependency
    grid = (1,)
    s111_kernel[grid](a, b, N)