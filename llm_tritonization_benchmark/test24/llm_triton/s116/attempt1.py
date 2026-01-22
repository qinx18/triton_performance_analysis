import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must process elements sequentially due to RAW dependencies
    # Use a single thread to process all elements in order
    pid = tl.program_id(0)
    
    if pid == 0:  # Only first thread block processes
        # Process elements sequentially in groups of 5
        for base_idx in range(0, n - 5, 5):
            # Load values from read-only copy
            a0_copy = tl.load(a_copy_ptr + base_idx)
            a1_copy = tl.load(a_copy_ptr + base_idx + 1)
            a2_copy = tl.load(a_copy_ptr + base_idx + 2)
            a3_copy = tl.load(a_copy_ptr + base_idx + 3)
            a4_copy = tl.load(a_copy_ptr + base_idx + 4)
            a5_copy = tl.load(a_copy_ptr + base_idx + 5)
            
            # Compute updates (note: these create dependencies)
            # a[i] = a[i + 1] * a[i]
            new_a0 = a1_copy * a0_copy
            tl.store(a_ptr + base_idx, new_a0)
            
            # a[i + 1] = a[i + 2] * a[i + 1] (but a[i+1] was just updated)
            updated_a1 = tl.load(a_ptr + base_idx + 1)  # Get current value
            new_a1 = a2_copy * updated_a1
            tl.store(a_ptr + base_idx + 1, new_a1)
            
            # a[i + 2] = a[i + 3] * a[i + 2] (but a[i+2] was just updated)  
            updated_a2 = tl.load(a_ptr + base_idx + 2)
            new_a2 = a3_copy * updated_a2
            tl.store(a_ptr + base_idx + 2, new_a2)
            
            # a[i + 3] = a[i + 4] * a[i + 3] (but a[i+3] was just updated)
            updated_a3 = tl.load(a_ptr + base_idx + 3)
            new_a3 = a4_copy * updated_a3
            tl.store(a_ptr + base_idx + 3, new_a3)
            
            # a[i + 4] = a[i + 5] * a[i + 4] (but a[i+4] was just updated)
            updated_a4 = tl.load(a_ptr + base_idx + 4)
            new_a4 = a5_copy * updated_a4
            tl.store(a_ptr + base_idx + 4, new_a4)

def s116_triton(a):
    n = a.shape[0]
    
    # Create read-only copy for WAR race condition handling
    a_copy = a.clone()
    
    # Use single thread block since processing must be sequential
    BLOCK_SIZE = 256
    grid = (1,)  # Only one thread block needed
    
    s116_kernel[grid](
        a, 
        a_copy, 
        n, 
        BLOCK_SIZE=BLOCK_SIZE
    )