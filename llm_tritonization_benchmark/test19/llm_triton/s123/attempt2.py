import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n):
    # Process each element sequentially since j depends on conditional updates
    j = -1
    for i in range(n):
        j = j + 1
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        b_val = tl.load(b_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        result1 = b_val + d_val * e_val
        tl.store(a_ptr + j, result1)
        
        # Check condition and conditionally update
        c_val = tl.load(c_ptr + i)
        if c_val > 0.0:
            j = j + 1
            result2 = c_val + d_val * e_val
            tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n = b.shape[0] // 2  # LEN_1D/2
    
    grid = (1,)  # Single thread block since we need sequential processing
    
    s123_kernel[grid](
        a, b, c, d, e,
        n
    )