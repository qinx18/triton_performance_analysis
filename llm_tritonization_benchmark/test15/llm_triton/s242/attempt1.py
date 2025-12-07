import triton
import triton.language as tl
import torch

@triton.jit
def s242_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s1, s2, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must run sequentially due to loop-carried dependency
    # Use single thread to process all elements sequentially
    thread_id = tl.program_id(0)
    
    # Only thread 0 processes the entire array sequentially
    if thread_id == 0:
        for i in range(1, n_elements):
            # Load a[i-1]
            prev_a = tl.load(a_ptr + i - 1)
            
            # Load b[i], c[i], d[i]
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            # Compute a[i] = a[i-1] + s1 + s2 + b[i] + c[i] + d[i]
            result = prev_a + s1 + s2 + b_val + c_val + d_val
            
            # Store a[i]
            tl.store(a_ptr + i, result)

def s242_triton(a, b, c, d, s1, s2):
    n_elements = a.shape[0]
    BLOCK_SIZE = 128
    
    # Launch with only 1 thread since computation must be sequential
    grid = (1,)
    
    s242_kernel[grid](
        a, b, c, d, 
        s1, s2, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a