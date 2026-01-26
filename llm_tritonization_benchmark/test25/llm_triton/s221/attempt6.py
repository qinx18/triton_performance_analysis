import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n):
    # Single thread processes all elements sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process sequentially from i=1 to n-1
    for i in range(1, n):
        # Load values for current iteration
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
    n = a.shape[0]
    
    # Launch with single thread due to sequential dependency
    grid = (1,)
    s221_kernel[grid](a, b, c, d, n)
    
    return a, b