import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, N, i_start):
    # Single thread processing one element at a time
    pid = tl.program_id(0)
    i = i_start + pid
    
    if i >= N:
        return
    
    # Load scalar values
    a_curr = tl.load(a_ptr + i)
    a_prev1 = tl.load(a_ptr + i - 1)
    a_prev2 = tl.load(a_ptr + i - 2)
    b_curr = tl.load(b_ptr + i)
    c_curr = tl.load(c_ptr + i)
    
    # Compute new value
    new_val = a_curr + a_prev1 * b_curr + a_prev2 * c_curr
    
    # Store result
    tl.store(a_ptr + i, new_val)

def s322_triton(a, b, c):
    N = a.shape[0]
    
    if N <= 2:
        return a
    
    # Process one element at a time to maintain dependencies
    for i in range(2, N):
        grid = (1,)
        s322_kernel[grid](a, b, c, N, i)
    
    return a