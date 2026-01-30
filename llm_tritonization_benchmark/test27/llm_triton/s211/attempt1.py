import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Sequential processing - no parallelization due to loop-carried dependency
    pid = tl.program_id(0)
    
    # Only use first thread
    if pid != 0:
        return
    
    # Process sequentially from i=1 to N-2
    for i in range(1, N - 1):
        # Load scalars
        b_prev = tl.load(b_ptr + (i - 1))
        b_next = tl.load(b_ptr + (i + 1))
        c_i = tl.load(c_ptr + i)
        d_i = tl.load(d_ptr + i)
        e_i = tl.load(e_ptr + i)
        
        # Compute and store
        a_val = b_prev + c_i * d_i
        b_val = b_next - e_i * d_i
        
        tl.store(a_ptr + i, a_val)
        tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Create read-only copy to handle WAR dependency
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single thread grid due to sequential nature
    
    s211_kernel[grid](
        a, b_copy, c, d, e,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )