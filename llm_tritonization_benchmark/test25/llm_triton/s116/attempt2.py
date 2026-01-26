import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements):
    # This kernel must be strictly sequential due to loop-carried dependencies
    # Use single thread to process all elements sequentially
    
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    # Process elements sequentially in groups of 5 (matching original unrolling)
    i = 0
    while i < n_elements:
        # Load values from read-only copy
        a_i = tl.load(a_copy_ptr + i)
        a_i_1 = tl.load(a_copy_ptr + i + 1)
        a_i_2 = tl.load(a_copy_ptr + i + 2)
        a_i_3 = tl.load(a_copy_ptr + i + 3)
        a_i_4 = tl.load(a_copy_ptr + i + 4)
        a_i_5 = tl.load(a_copy_ptr + i + 5)
        
        # Compute new values (sequential dependencies)
        new_a_i = a_i_1 * a_i
        tl.store(a_ptr + i, new_a_i)
        
        new_a_i_1 = a_i_2 * new_a_i
        tl.store(a_ptr + i + 1, new_a_i_1)
        
        new_a_i_2 = a_i_3 * new_a_i_1
        tl.store(a_ptr + i + 2, new_a_i_2)
        
        new_a_i_3 = a_i_4 * new_a_i_2
        tl.store(a_ptr + i + 3, new_a_i_3)
        
        new_a_i_4 = a_i_5 * new_a_i_3
        tl.store(a_ptr + i + 4, new_a_i_4)
        
        i += 5

def s116_triton(a):
    N = a.shape[0]
    n_elements = N - 5
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    # Use single thread due to sequential dependencies
    grid = (1,)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements
    )
    
    return a