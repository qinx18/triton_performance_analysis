import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements):
    # This kernel must process sequentially due to loop-carried dependency
    # b[i] depends on b[i-1] from previous iteration
    
    # Process all elements sequentially in a single thread
    if tl.program_id(0) == 0:
        for i in range(1, n_elements - 1):
            # Load values
            b_prev = tl.load(b_ptr + (i - 1))
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            # First statement: a[i] = b[i-1] + c[i]
            a_val = b_prev + c_val
            tl.store(a_ptr + i, a_val)
            
            # Second statement: b[i] = a[i+1] * d[i]
            # Need to load a[i+1] which may have been updated
            a_next = tl.load(a_ptr + (i + 1))
            b_val = a_next * d_val
            tl.store(b_ptr + i, b_val)

def s1213_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Use single thread since this must be sequential
    grid = (1,)
    
    s1213_kernel[grid](
        a, b, c, d, n_elements
    )