import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must process sequentially due to RAW dependency
    # Use single thread to process all elements
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process sequentially starting from i=1, step=2
    i = 1
    while i < n_elements:
        # Load a[i-1] and b[i]
        a_prev = tl.load(a_ptr + (i - 1))
        b_val = tl.load(b_ptr + i)
        
        # Compute a[i] = a[i-1] + b[i]
        result = a_prev + b_val
        
        # Store result
        tl.store(a_ptr + i, result)
        
        i += 2

def s111_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch single thread since computation is strictly sequential
    grid = (1,)
    
    s111_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )