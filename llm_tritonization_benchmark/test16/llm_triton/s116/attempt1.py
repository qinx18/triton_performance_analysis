import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n):
    # This kernel must be strictly sequential due to loop-carried dependencies
    # Use a single thread to process all elements sequentially
    
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially in groups of 5
    for i in range(0, n - 5, 5):
        # Load values needed for this group of 5 operations
        a_i = tl.load(a_copy_ptr + i)
        a_i1 = tl.load(a_copy_ptr + i + 1)
        a_i2 = tl.load(a_copy_ptr + i + 2)
        a_i3 = tl.load(a_copy_ptr + i + 3)
        a_i4 = tl.load(a_copy_ptr + i + 4)
        a_i5 = tl.load(a_copy_ptr + i + 5)
        
        # Compute and store sequentially to maintain dependencies
        result_0 = a_i1 * a_i
        tl.store(a_ptr + i, result_0)
        
        result_1 = a_i2 * result_0
        tl.store(a_ptr + i + 1, result_1)
        
        result_2 = a_i3 * result_1
        tl.store(a_ptr + i + 2, result_2)
        
        result_3 = a_i4 * result_2
        tl.store(a_ptr + i + 3, result_3)
        
        result_4 = a_i5 * result_3
        tl.store(a_ptr + i + 4, result_4)

def s116_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Use single thread due to strict sequential requirements
    grid = (1,)
    
    s116_kernel[grid](
        a,
        a_copy,
        n
    )
    
    return a