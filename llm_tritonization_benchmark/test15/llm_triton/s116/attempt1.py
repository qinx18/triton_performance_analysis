import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements):
    # This kernel must process elements sequentially due to RAW dependencies
    # Use a single thread to process all elements
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Process elements sequentially in groups of 5
    i = 0
    while i < n_elements - 5:
        # Load values from read-only copy
        a_i = tl.load(a_copy_ptr + i)
        a_i1 = tl.load(a_copy_ptr + i + 1)
        a_i2 = tl.load(a_copy_ptr + i + 2)
        a_i3 = tl.load(a_copy_ptr + i + 3)
        a_i4 = tl.load(a_copy_ptr + i + 4)
        a_i5 = tl.load(a_copy_ptr + i + 5)
        
        # Compute new values
        new_a_i = a_i1 * a_i
        new_a_i1 = a_i2 * a_i1
        new_a_i2 = a_i3 * a_i2
        new_a_i3 = a_i4 * a_i3
        new_a_i4 = a_i5 * a_i4
        
        # Store to original array
        tl.store(a_ptr + i, new_a_i)
        tl.store(a_ptr + i + 1, new_a_i1)
        tl.store(a_ptr + i + 2, new_a_i2)
        tl.store(a_ptr + i + 3, new_a_i3)
        tl.store(a_ptr + i + 4, new_a_i4)
        
        # Update copy for next iteration (sequential dependency)
        tl.store(a_copy_ptr + i, new_a_i)
        tl.store(a_copy_ptr + i + 1, new_a_i1)
        tl.store(a_copy_ptr + i + 2, new_a_i2)
        tl.store(a_copy_ptr + i + 3, new_a_i3)
        tl.store(a_copy_ptr + i + 4, new_a_i4)
        
        i += 5

def s116_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    # Use single thread due to sequential dependencies
    grid = (1,)
    
    s116_kernel[grid](
        a, a_copy, n_elements
    )