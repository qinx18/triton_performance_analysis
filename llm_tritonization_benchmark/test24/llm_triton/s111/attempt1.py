import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must process strictly sequentially due to RAW dependency
    # Use single thread to ensure correct ordering
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process all odd indices sequentially
    for i in range(1, n, 2):
        # Load a[i-1] and b[i]
        a_prev = tl.load(a_ptr + (i - 1))
        b_val = tl.load(b_ptr + i)
        
        # Compute and store a[i] = a[i-1] + b[i]
        result = a_prev + b_val
        tl.store(a_ptr + i, result)

def s111_triton(a, b):
    n = a.shape[0]
    
    # Use single thread since computation must be strictly sequential
    grid = (1,)
    BLOCK_SIZE = 256
    
    s111_kernel[grid](a, b, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return a