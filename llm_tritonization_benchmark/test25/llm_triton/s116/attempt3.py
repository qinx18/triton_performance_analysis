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
    
    # Process elements sequentially one by one
    i = 0
    while i < n_elements:
        # Load current value and next value from read-only copy
        a_i = tl.load(a_copy_ptr + i)
        a_i_plus_1 = tl.load(a_copy_ptr + i + 1)
        
        # Compute: a[i] = a[i + 1] * a[i]
        new_val = a_i_plus_1 * a_i
        
        # Store the result
        tl.store(a_ptr + i, new_val)
        
        i += 1

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