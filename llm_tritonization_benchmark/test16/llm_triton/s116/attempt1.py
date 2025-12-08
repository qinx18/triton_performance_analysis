import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependencies
    # Use a single thread to process all elements sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially in groups of 5
    for block_start in range(0, n_elements - 5, 5):
        # Load values needed for this group of 5 operations
        val_0 = tl.load(a_copy_ptr + block_start)
        val_1 = tl.load(a_copy_ptr + block_start + 1)
        val_2 = tl.load(a_copy_ptr + block_start + 2)
        val_3 = tl.load(a_copy_ptr + block_start + 3)
        val_4 = tl.load(a_copy_ptr + block_start + 4)
        val_5 = tl.load(a_copy_ptr + block_start + 5)
        
        # Perform the 5 operations sequentially
        new_val_0 = val_1 * val_0
        new_val_1 = val_2 * val_1
        new_val_2 = val_3 * val_2
        new_val_3 = val_4 * val_3
        new_val_4 = val_5 * val_4
        
        # Store results
        tl.store(a_ptr + block_start, new_val_0)
        tl.store(a_ptr + block_start + 1, new_val_1)
        tl.store(a_ptr + block_start + 2, new_val_2)
        tl.store(a_ptr + block_start + 3, new_val_3)
        tl.store(a_ptr + block_start + 4, new_val_4)

def s116_triton(a):
    n_elements = a.size(0)
    BLOCK_SIZE = 256
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Use single thread since this must be sequential
    grid = (1,)
    
    s116_kernel[grid](
        a,
        a_copy,
        n_elements,
        BLOCK_SIZE
    )