import triton
import triton.language as tl
import torch

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependency
    # b[i] depends on b[i-1] from previous iteration
    
    # Use single thread to process sequentially
    thread_id = tl.program_id(0)
    if thread_id != 0:
        return
    
    # Process all elements sequentially
    for i in range(1, N - 1):
        # Load values
        b_prev = tl.load(b_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        a_next = tl.load(a_ptr + i + 1)
        d_val = tl.load(d_ptr + i)
        
        # Compute and store a[i]
        a_val = b_prev + c_val
        tl.store(a_ptr + i, a_val)
        
        # Compute and store b[i]
        b_val = a_next * d_val
        tl.store(b_ptr + i, b_val)

def s1213_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single thread since computation must be sequential
    grid = (1,)
    s1213_kernel[grid](
        a, b, c, d, N, BLOCK_SIZE=BLOCK_SIZE
    )