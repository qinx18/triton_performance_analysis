import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements):
    BLOCK_SIZE = 1
    pid = tl.program_id(0)
    
    if pid == 0:
        # Process all elements sequentially in a single thread
        for i in range(0, n_elements - 5, 5):
            # Load current values from read-only copy
            a_i = tl.load(a_copy_ptr + i)
            a_i1 = tl.load(a_copy_ptr + i + 1)
            a_i2 = tl.load(a_copy_ptr + i + 2)
            a_i3 = tl.load(a_copy_ptr + i + 3)
            a_i4 = tl.load(a_copy_ptr + i + 4)
            a_i5 = tl.load(a_copy_ptr + i + 5)
            
            # Compute updates sequentially
            new_a_i = a_i1 * a_i
            tl.store(a_ptr + i, new_a_i)
            
            new_a_i1 = a_i2 * new_a_i
            tl.store(a_ptr + i + 1, new_a_i1)
            
            new_a_i2 = a_i3 * new_a_i1
            tl.store(a_ptr + i + 2, new_a_i2)
            
            new_a_i3 = a_i4 * new_a_i2
            tl.store(a_ptr + i + 3, new_a_i3)
            
            new_a_i4 = a_i5 * new_a_i3
            tl.store(a_ptr + i + 4, new_a_i4)

def s116_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy for safe concurrent reads
    a_copy = a.clone()
    
    # Launch single thread to handle sequential computation
    grid = (1,)
    s116_kernel[grid](a, a_copy, n_elements)
    
    return a