import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependencies
    # Use single thread to process all elements sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially in groups of 5 (unrolled)
    for block_start in range(0, n_elements - 5, 5):
        # Load values from read-only copy
        val0 = tl.load(a_copy_ptr + block_start)
        val1 = tl.load(a_copy_ptr + block_start + 1)
        val2 = tl.load(a_copy_ptr + block_start + 2)
        val3 = tl.load(a_copy_ptr + block_start + 3)
        val4 = tl.load(a_copy_ptr + block_start + 4)
        val5 = tl.load(a_copy_ptr + block_start + 5)
        
        # Compute new values
        new_val0 = val1 * val0
        new_val1 = val2 * val1
        new_val2 = val3 * val2
        new_val3 = val4 * val3
        new_val4 = val5 * val4
        
        # Store to original array
        tl.store(a_ptr + block_start, new_val0)
        tl.store(a_ptr + block_start + 1, new_val1)
        tl.store(a_ptr + block_start + 2, new_val2)
        tl.store(a_ptr + block_start + 3, new_val3)
        tl.store(a_ptr + block_start + 4, new_val4)

def s116_triton(a):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single thread block for sequential processing
    
    s116_kernel[grid](
        a, a_copy, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )