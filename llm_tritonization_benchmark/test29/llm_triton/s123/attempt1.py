import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half):
    # Process sequentially since j depends on conditional increment
    j = -1
    
    for i in range(n_half):
        j += 1
        
        # Load values for current iteration
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        result1 = b_val + d_val * e_val
        tl.store(a_ptr + j, result1)
        
        # Conditional assignment
        if c_val > 0.0:
            j += 1
            result2 = c_val + d_val * e_val
            tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n = b.shape[0]
    n_half = n // 2
    
    # Launch single thread kernel since we need sequential processing
    grid = (1,)
    s123_kernel[grid](a, b, c, d, e, n_half)
    
    return a