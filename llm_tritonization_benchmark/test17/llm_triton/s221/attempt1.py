import torch
import triton
import triton.language as tl

@triton.jit
def s221_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Single thread processes all elements sequentially
    thread_id = tl.program_id(axis=0)
    
    # Only one thread should run
    if thread_id != 0:
        return
    
    # Process elements sequentially from index 1
    for i in range(1, n_elements):
        # Load c[i] and d[i]
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        # Load a[i]
        a_val = tl.load(a_ptr + i)
        
        # Update a[i] += c[i] * d[i]
        a_new = a_val + c_val * d_val
        tl.store(a_ptr + i, a_new)
        
        # Load b[i-1]
        b_prev = tl.load(b_ptr + i - 1)
        
        # Compute b[i] = b[i-1] + a[i] + d[i]
        b_new = b_prev + a_new + d_val
        tl.store(b_ptr + i, b_new)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Use single thread due to strict sequential dependency
    grid = (1,)
    BLOCK_SIZE = 1024
    
    s221_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE,
    )
    
    return a, b