import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n):
    # This kernel must be strictly sequential due to loop-carried dependency
    # Use a single program with sequential processing
    
    for i in range(1, n - 1):
        # a[i] = b[i-1] + c[i]
        b_val = tl.load(b_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        a_val = b_val + c_val
        tl.store(a_ptr + i, a_val)
        
        # b[i] = a[i+1] * d[i] (using original a values)
        a_next_val = tl.load(a_copy_ptr + i + 1)
        d_val = tl.load(d_ptr + i)
        b_val = a_next_val * d_val
        tl.store(b_ptr + i, b_val)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    # Use single thread for sequential processing
    grid = (1,)
    
    s1213_kernel[grid](
        a, a_copy, b, c, d, n
    )