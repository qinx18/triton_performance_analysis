import triton
import triton.language as tl
import torch

@triton.jit
def s342_kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program processes one element at a time due to sequential dependency
    idx = pid
    
    if idx >= n:
        return
        
    # Load a[idx]
    a_val = tl.load(a_ptr + idx)
    
    # Check condition
    if a_val > 0.0:
        # Count how many positive elements come before this index
        j = -1
        for i in range(idx + 1):
            if i < n:
                prev_a_val = tl.load(a_ptr + i)
                if prev_a_val > 0.0:
                    j = j + 1
        
        # Load b[j] and store to a[idx]
        if j >= 0:
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + idx, b_val)

def s342_triton(a, b):
    n = a.shape[0]
    
    # Launch one thread per element since we need sequential dependency handling
    grid = (n,)
    
    s342_kernel[grid](
        a, b, n,
        BLOCK_SIZE=1
    )