import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependencies
    # Use single thread to process all elements sequentially
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Process elements sequentially one by one
    for i in range(n - 1):
        # Load from read-only copy for the read operand
        a_read = tl.load(a_copy_ptr + i + 1)
        # Load current value from original array for the other operand
        a_current = tl.load(a_ptr + i)
        
        # Compute new value: a[i] = a[i + 1] * a[i]
        new_val = a_read * a_current
        
        # Store back to original array
        tl.store(a_ptr + i, new_val)

def s116_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Use single thread since computation must be sequential
    grid = (1,)
    BLOCK_SIZE = 256
    
    s116_kernel[grid](
        a, a_copy, n, 
        BLOCK_SIZE=BLOCK_SIZE
    )