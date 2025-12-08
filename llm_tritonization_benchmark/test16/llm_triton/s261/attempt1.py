import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must process sequentially due to loop-carried dependency
    # Use single thread to maintain sequential order
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process sequentially from i=1 to n_elements-1
    for i in range(1, n_elements):
        # Load values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_prev = tl.load(c_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Compute
        t = a_val + b_val
        new_a = t + c_prev
        t = c_val * d_val
        
        # Store results
        tl.store(a_ptr + i, new_a)
        tl.store(c_ptr + i, t)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single block since we need sequential processing
    grid = (1,)
    
    s261_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, c