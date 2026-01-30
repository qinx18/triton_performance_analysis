import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n):
    # This kernel must be strictly sequential due to RAW dependencies
    # Use a single thread to process all elements
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Process elements sequentially in groups of 5
    for block_start in range(0, n - 5, 5):
        # Load values for the 5-element block
        a0 = tl.load(a_copy_ptr + block_start)
        a1 = tl.load(a_copy_ptr + block_start + 1)
        a2 = tl.load(a_copy_ptr + block_start + 2)
        a3 = tl.load(a_copy_ptr + block_start + 3)
        a4 = tl.load(a_copy_ptr + block_start + 4)
        a5 = tl.load(a_copy_ptr + block_start + 5)
        
        # Compute the 5 operations
        result0 = a1 * a0
        result1 = a2 * a1
        result2 = a3 * a2
        result3 = a4 * a3
        result4 = a5 * a4
        
        # Store results
        tl.store(a_ptr + block_start, result0)
        tl.store(a_ptr + block_start + 1, result1)
        tl.store(a_ptr + block_start + 2, result2)
        tl.store(a_ptr + block_start + 3, result3)
        tl.store(a_ptr + block_start + 4, result4)

def s116_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Use single thread grid since computation must be sequential
    grid = (1,)
    
    s116_kernel[grid](
        a,
        a_copy,
        n
    )