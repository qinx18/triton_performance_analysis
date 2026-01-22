import triton
import triton.language as tl

@triton.jit
def s3251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n):
    # This kernel must run sequentially due to loop-carried dependency
    # Use only one thread to process all elements
    thread_id = tl.program_id(0)
    
    if thread_id > 0:
        return
    
    # Process all iterations sequentially
    for i in range(n - 1):
        # Load values for current iteration
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        e_val = tl.load(e_ptr + i)
        a_val = tl.load(a_ptr + i)
        
        # Compute and store results
        a_new = b_val + c_val
        b_new = c_val * e_val
        d_new = a_val * e_val
        
        tl.store(a_ptr + i + 1, a_new)
        tl.store(b_ptr + i, b_new)
        tl.store(d_ptr + i, d_new)

def s3251_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Launch with single thread due to sequential dependency
    grid = (1,)
    s3251_kernel[grid](a, b, c, d, e, n)