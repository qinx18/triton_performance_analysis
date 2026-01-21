import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried RAW dependency
    # b[i] depends on b[i-1] from previous iteration
    
    # Use single thread to process sequentially
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Sequential processing from i=1 to N-2
    for i in range(1, N-1):
        # Load values for current iteration
        b_prev = tl.load(b_ptr + (i - 1))
        b_next = tl.load(b_ptr + (i + 1))
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Compute and store results
        a_val = b_prev + c_val * d_val
        b_val = b_next - e_val * d_val
        
        tl.store(a_ptr + i, a_val)
        tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Use single thread since computation must be sequential
    grid = (1,)
    BLOCK_SIZE = 256
    
    s211_kernel[grid](
        a, b, c, d, e, N, BLOCK_SIZE=BLOCK_SIZE
    )