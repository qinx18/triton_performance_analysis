import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements):
    # This kernel must be strictly sequential due to loop-carried dependency
    # b[i] depends on b[i-1] from previous iteration
    
    # Use single thread to process sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process iterations 1 to n_elements-2 (equivalent to i=1; i<LEN_1D-1)
    for i in range(1, n_elements - 1):
        # Load values
        b_prev = tl.load(b_ptr + (i - 1))
        c_curr = tl.load(c_ptr + i)
        a_next = tl.load(a_ptr + (i + 1))
        d_curr = tl.load(d_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i]
        a_val = b_prev + c_curr
        tl.store(a_ptr + i, a_val)
        
        # Compute b[i] = a[i+1] * d[i]
        b_val = a_next * d_curr
        tl.store(b_ptr + i, b_val)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    
    s1213_kernel[grid](
        a, b, c, d, n_elements
    )