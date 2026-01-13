import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, N):
    # This kernel must run sequentially due to loop-carried dependency
    # Use single thread to process all elements
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Sequential processing from i=1 to N-2
    for i in range(1, N-1):
        # Load values
        b_prev = tl.load(b_copy_ptr + i - 1)
        b_next = tl.load(b_copy_ptr + i + 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Compute
        a_val = b_prev + c_val * d_val
        b_val = b_next - e_val * d_val
        
        # Store
        tl.store(a_ptr + i, a_val)
        tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Create read-only copy of b to handle WAR dependency
    b_copy = b.clone()
    
    # Launch single thread to handle sequential dependency
    grid = (1,)
    s211_kernel[grid](
        a, b, b_copy, c, d, e, N
    )