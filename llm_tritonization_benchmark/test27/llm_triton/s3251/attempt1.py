import triton
import triton.language as tl
import torch

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n):
    # This kernel must be sequential due to RAW dependency: a[i+1] = f(a[i])
    # Use single thread to process all elements sequentially
    pid = tl.program_id(0)
    
    # Only thread 0 processes the entire loop sequentially
    if pid == 0:
        for i in range(n - 1):
            # Load values for current iteration
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            e_val = tl.load(e_ptr + i)
            a_val = tl.load(a_ptr + i)
            
            # Compute updates
            new_a = b_val + c_val
            new_b = c_val * e_val
            new_d = a_val * e_val
            
            # Store results
            tl.store(a_ptr + i + 1, new_a)
            tl.store(b_ptr + i, new_b)
            tl.store(d_ptr + i, new_d)

def s3251_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    
    s3251_kernel[grid](
        a, b, c, d, e, n
    )