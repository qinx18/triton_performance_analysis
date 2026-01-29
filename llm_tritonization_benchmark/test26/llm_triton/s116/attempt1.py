import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n):
    # Single thread processing - no parallelization due to sequential dependencies
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Sequential processing of all elements
    for base_i in range(0, n - 5, 5):
        # Process 5 consecutive elements sequentially
        for offset in range(5):
            i = base_i + offset
            
            # Load values needed for computation
            a_i = tl.load(a_copy_ptr + i)
            a_i_plus_1 = tl.load(a_copy_ptr + i + 1)
            
            # Compute and store
            result = a_i_plus_1 * a_i
            tl.store(a_ptr + i, result)

def s116_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Use single thread due to sequential dependencies
    grid = (1,)
    
    s116_kernel[grid](
        a,
        a_copy,
        n
    )