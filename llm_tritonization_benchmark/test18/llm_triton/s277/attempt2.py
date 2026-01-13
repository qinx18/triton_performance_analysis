import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n):
    # This kernel must process ALL iterations sequentially due to RAW dependency
    # b[i+1] depends on b[i] from previous iteration
    
    for i in range(n - 1):
        # Load values for current iteration
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Check if a[i] >= 0, if so, skip to L20 (end of iteration)
        a_condition = a_val >= 0.0
        
        # Check if b[i] >= 0, if so, skip to L30
        b_condition = b_val >= 0.0
        
        # Execute a[i] += c[i] * d[i] only if both conditions are false
        execute_a_update = (a_condition == 0) & (b_condition == 0)
        if execute_a_update:
            a_new_val = a_val + c_val * d_val
            tl.store(a_ptr + i, a_new_val)
        
        # Execute L30: b[i+1] = c[i] + d[i] * e[i] unless a[i] >= 0
        execute_b_update = a_condition == 0
        if execute_b_update:
            b_next_val = c_val + d_val * e_val
            tl.store(b_ptr + i + 1, b_next_val)

def s277_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Launch single thread since we must process sequentially
    grid = (1,)
    s277_kernel[grid](a, b, c, d, e, n)