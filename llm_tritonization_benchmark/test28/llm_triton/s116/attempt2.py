import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n):
    # Process strictly sequentially due to RAW dependencies
    # Use single thread processing all elements in order
    
    for i in range(n - 5):
        if i % 5 == 0:
            # Load the initial values for this group of 5
            val_0 = tl.load(a_copy_ptr + i)
            val_1 = tl.load(a_copy_ptr + i + 1)
            val_2 = tl.load(a_copy_ptr + i + 2)
            val_3 = tl.load(a_copy_ptr + i + 3)
            val_4 = tl.load(a_copy_ptr + i + 4)
            val_5 = tl.load(a_copy_ptr + i + 5)
            
            # Apply the dependency chain
            new_0 = val_1 * val_0
            new_1 = val_2 * new_0  # Use updated value
            new_2 = val_3 * new_1  # Use updated value
            new_3 = val_4 * new_2  # Use updated value
            new_4 = val_5 * new_3  # Use updated value
            
            # Store all 5 results
            tl.store(a_ptr + i, new_0)
            tl.store(a_ptr + i + 1, new_1)
            tl.store(a_ptr + i + 2, new_2)
            tl.store(a_ptr + i + 3, new_3)
            tl.store(a_ptr + i + 4, new_4)

def s116_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to handle dependencies
    a_copy = a.clone()
    
    # Use single thread for strictly sequential execution
    grid = (1,)
    
    s116_kernel[grid](
        a,
        a_copy,
        n
    )