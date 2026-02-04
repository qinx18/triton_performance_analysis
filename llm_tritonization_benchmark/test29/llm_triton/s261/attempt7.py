import triton
import triton.language as tl
import torch

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel processes sequentially due to loop-carried dependency
    # Only one thread block processes the entire array
    pid = tl.program_id(0)
    
    if pid == 0:
        # Process elements sequentially from 1 to n-1
        for i in range(1, n):
            # Load scalar values
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_prev = tl.load(c_ptr + i - 1)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            # Compute
            t = a_val + b_val
            new_a = t + c_prev
            t = c_val * d_val
            new_c = t
            
            # Store results
            tl.store(a_ptr + i, new_a)
            tl.store(c_ptr + i, new_c)

def s261_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Use single thread block due to loop-carried dependency
    grid = (1,)
    
    s261_kernel[grid](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )