import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n):
    # This kernel must run sequentially due to loop-carried dependency
    # Use a single thread to process all elements
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process sequentially from i=1 to n-1, step=2
    for i in range(1, n, 2):
        # Load a[i-1] and b[i]
        a_prev = tl.load(a_ptr + i - 1)
        b_val = tl.load(b_ptr + i)
        
        # Compute and store a[i] = a[i-1] + b[i]
        result = a_prev + b_val
        tl.store(a_ptr + i, result)

def s111_triton(a, b):
    n = a.shape[0]
    
    # Launch with single thread due to sequential dependency
    grid = (1,)
    s111_kernel[grid](a, b, n)