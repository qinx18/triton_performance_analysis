import triton
import triton.language as tl

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements):
    # This kernel must run with a single thread due to strict sequential dependency
    # b[i] depends on b[i-1], so no parallelization is possible
    
    # Process all elements sequentially in a single thread
    for i in range(1, n_elements):
        # Load values
        a_val = tl.load(a_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        b_prev = tl.load(b_ptr + i - 1)
        
        # Compute a[i] += c[i] * d[i]
        new_a = a_val + c_val * d_val
        tl.store(a_ptr + i, new_a)
        
        # Compute b[i] = b[i-1] + a[i] + d[i]
        new_b = b_prev + new_a + d_val
        tl.store(b_ptr + i, new_b)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Launch with single thread since computation is strictly sequential
    s221_kernel[(1,)](
        a, b, c, d, n_elements
    )