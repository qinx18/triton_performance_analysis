import torch
import triton
import triton.language as tl

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n):
    # This kernel must process the entire array sequentially due to RAW dependencies
    # Each element depends on the two previous elements, so no parallelization is possible
    
    # Use only one thread to process the entire computation sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process sequentially from index 2 to n-1
    for i in range(2, n):
        # Load current values
        a_i = tl.load(a_ptr + i)
        a_i_minus_1 = tl.load(a_ptr + i - 1)
        a_i_minus_2 = tl.load(a_ptr + i - 2)
        b_i = tl.load(b_ptr + i)
        c_i = tl.load(c_ptr + i)
        
        # Compute new value
        new_a_i = a_i + a_i_minus_1 * b_i + a_i_minus_2 * c_i
        
        # Store result
        tl.store(a_ptr + i, new_a_i)

def s322_triton(a, b, c):
    n = a.shape[0]
    
    # Launch with only one thread block since computation must be sequential
    grid = (1,)
    s322_kernel[grid](a, b, c, n)
    
    return a