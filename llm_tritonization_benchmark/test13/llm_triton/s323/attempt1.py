import torch
import triton
import triton.language as tl

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must process sequentially due to loop-carried dependency
    # Use single thread to process all elements
    thread_id = tl.program_id(0)
    
    if thread_id != 0:
        return
    
    # Process sequentially from i=1 to n_elements-1
    for i in range(1, n_elements):
        # Load required values
        b_prev = tl.load(b_ptr + (i - 1))
        c_i = tl.load(c_ptr + i)
        d_i = tl.load(d_ptr + i)
        e_i = tl.load(e_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_val = b_prev + c_i * d_i
        tl.store(a_ptr + i, a_val)
        
        # Compute b[i] = a[i] + c[i] * e[i]
        b_val = a_val + c_i * e_i
        tl.store(b_ptr + i, b_val)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 128
    
    # Launch with single block since we need sequential processing
    grid = (1,)
    
    s323_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )