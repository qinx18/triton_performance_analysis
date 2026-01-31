import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n):
    # This kernel must be strictly sequential due to RAW dependencies
    # Use a single thread to process all elements sequentially
    
    offsets = tl.arange(0, 1024)  # Define once at start
    
    # Process elements sequentially in chunks
    for block_start in range(0, n - 5, 1024):
        block_end = min(block_start + 1024, n - 5)
        
        for local_i in range(block_end - block_start):
            i = block_start + local_i
            
            # Each iteration processes 5 consecutive elements
            if i + 5 < n:
                # Load values from read-only copy
                val_i = tl.load(a_copy_ptr + i)
                val_i1 = tl.load(a_copy_ptr + i + 1)
                val_i2 = tl.load(a_copy_ptr + i + 2)
                val_i3 = tl.load(a_copy_ptr + i + 3)
                val_i4 = tl.load(a_copy_ptr + i + 4)
                val_i5 = tl.load(a_copy_ptr + i + 5)
                
                # Compute new values following the dependency chain
                new_val_i = val_i1 * val_i
                tl.store(a_ptr + i, new_val_i)
                
                # Update val_i1 to use the newly computed value
                new_val_i1 = val_i2 * new_val_i
                tl.store(a_ptr + i + 1, new_val_i1)
                
                new_val_i2 = val_i3 * new_val_i1
                tl.store(a_ptr + i + 2, new_val_i2)
                
                new_val_i3 = val_i4 * new_val_i2
                tl.store(a_ptr + i + 3, new_val_i3)
                
                new_val_i4 = val_i5 * new_val_i3
                tl.store(a_ptr + i + 4, new_val_i4)

def s116_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Use single thread for strictly sequential execution
    grid = (1,)
    
    s116_kernel[grid](
        a,
        a_copy,
        n
    )