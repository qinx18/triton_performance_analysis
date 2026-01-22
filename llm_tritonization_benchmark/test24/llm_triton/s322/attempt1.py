import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, N):
    # This kernel must process the entire array sequentially
    # due to the strict RAW dependency: a[i] depends on a[i-1]
    
    # Use a single thread to process all elements sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially from index 2 to N-1
    for i in range(2, N):
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
    N = a.shape[0]
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    s322_kernel[grid](a, b, c, N)
    
    return a