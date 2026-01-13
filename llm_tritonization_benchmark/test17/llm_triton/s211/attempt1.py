import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must be processed sequentially due to loop-carried dependency
    # Use single thread to process all elements sequentially
    
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Process sequentially from i=1 to n_elements-2
    for i in range(1, n_elements - 1):
        # Load values for current iteration
        b_prev = tl.load(b_copy_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        b_next = tl.load(b_copy_ptr + i + 1)
        e_val = tl.load(e_ptr + i)
        
        # Compute and store a[i]
        a_val = b_prev + c_val * d_val
        tl.store(a_ptr + i, a_val)
        
        # Compute and store b[i]
        b_val = b_next - e_val * d_val
        tl.store(b_ptr + i, b_val)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR safety
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single block since we need sequential processing
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )