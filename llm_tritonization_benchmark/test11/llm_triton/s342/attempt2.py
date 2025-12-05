import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This algorithm requires sequential processing due to j dependency
    # We'll process one element at a time to maintain correctness
    pid = tl.program_id(0)
    
    # Each program processes one element
    if pid < n_elements:
        # Count how many positive elements come before this position
        j = -1
        for i in range(pid + 1):
            a_val = tl.load(a_ptr + i)
            if a_val > 0.0:
                j += 1
        
        # Check if current element is positive
        a_current = tl.load(a_ptr + pid)
        if a_current > 0.0:
            # Load b[j] and store to a[pid]
            b_val = tl.load(b_ptr + j)
            tl.store(a_ptr + pid, b_val)

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch one thread per element for sequential correctness
    grid = (n_elements,)
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a