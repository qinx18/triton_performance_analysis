import triton
import triton.language as tl
import torch

@triton.jit
def s322_kernel(a_ptr, b_ptr, c_ptr, n_elements):
    # This kernel must run sequentially due to strict RAW dependencies
    # Use a single thread to process all elements
    thread_id = tl.program_id(0)
    
    # Only thread 0 processes the entire array sequentially
    if thread_id == 0:
        for i in range(2, n_elements):
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
    n_elements = a.shape[0]
    
    # Launch with only 1 thread since computation must be sequential
    grid = (1,)
    
    s322_kernel[grid](
        a, b, c, n_elements
    )
    
    return a