import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n):
    # This kernel must run sequentially due to strict dependencies
    # Use single thread to process all elements
    pid = tl.program_id(0)
    
    # Only the first thread does all the work
    if pid == 0:
        for i in range(2, n):
            # Load current values
            a_i = tl.load(a_ptr + i)
            a_i_minus_1 = tl.load(a_ptr + i - 1)
            a_i_minus_2 = tl.load(a_ptr + i - 2)
            b_i = tl.load(b_ptr + i)
            c_i = tl.load(c_ptr + i)
            
            # Compute new value
            new_val = a_i + a_i_minus_1 * b_i + a_i_minus_2 * c_i
            
            # Store result
            tl.store(a_ptr + i, new_val)

def s322_triton(a, b, c):
    n = a.shape[0]
    
    # Launch with single block, single thread due to sequential dependency
    grid = (1,)
    s322_kernel[grid](a, b, c, n)
    
    return a