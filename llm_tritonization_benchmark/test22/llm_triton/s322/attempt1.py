import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n):
    # This kernel must run sequentially due to loop-carried dependencies
    # Use a single thread to process all elements
    pid = tl.program_id(axis=0)
    if pid > 0:
        return
    
    # Process sequentially from i=2 to n-1
    for i in range(2, n):
        # Load current values
        a_i = tl.load(a_ptr + i)
        a_i_minus_1 = tl.load(a_ptr + i - 1)
        a_i_minus_2 = tl.load(a_ptr + i - 2)
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        
        # Compute: a[i] = a[i] + a[i-1] * b[i] + a[i-2] * c[i]
        result = a_i + a_i_minus_1 * b_i + a_i_minus_2 * c_i
        
        # Store result
        tl.store(a_ptr + i, result)

def s322_triton(a, b, c):
    n = a.shape[0]
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    s322_kernel[grid](a, b, c, n)
    
    return a