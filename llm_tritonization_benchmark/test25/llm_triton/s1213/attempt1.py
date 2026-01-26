import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements):
    # This kernel must be strictly sequential due to loop-carried dependency
    # b[i] depends on b[i-1] from previous iteration
    
    # Use single thread to process all elements sequentially
    thread_id = tl.program_id(0)
    if thread_id != 0:
        return
    
    # Process elements sequentially from 1 to n_elements-2
    for i in range(1, n_elements - 1):
        # Load values for current iteration
        b_prev = tl.load(b_ptr + (i - 1))
        c_curr = tl.load(c_ptr + i)
        a_next = tl.load(a_copy_ptr + (i + 1))
        d_curr = tl.load(d_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i]
        a_val = b_prev + c_curr
        tl.store(a_ptr + i, a_val)
        
        # Compute b[i] = a[i+1] * d[i]
        b_val = a_next * d_curr
        tl.store(b_ptr + i, b_val)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Create read-only copy of 'a' to handle WAR dependency
    a_copy = a.clone()
    
    # Use single thread since computation must be strictly sequential
    grid = (1,)
    
    s1213_kernel[grid](
        a, a_copy, b, c, d, n_elements
    )