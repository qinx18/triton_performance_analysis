import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, n_elements):
    # Single thread processes all elements sequentially
    if tl.program_id(0) != 0:
        return
    
    # Process elements sequentially from 1 to n_elements-2
    for i in range(1, n_elements - 1):
        # Load values for a[i] computation
        b_prev = tl.load(b_copy_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Load values for b[i] computation
        b_next = tl.load(b_copy_ptr + i + 1)
        e_val = tl.load(e_ptr + i)
        
        # Compute
        a_val = b_prev + c_val * d_val
        b_val = b_next - e_val * d_val
        
        # Store
        tl.store(a_ptr + i, a_val)
        tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR safety
    b_copy = b.clone()
    
    # Launch single thread for sequential processing
    grid = (1,)
    s211_kernel[grid](
        a, b, b_copy, c, d, e, n_elements
    )