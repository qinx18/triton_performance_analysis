import torch
import triton
import triton.language as tl

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n):
    # This kernel must be strictly sequential due to loop-carried dependencies
    # Use only one thread to process all elements sequentially
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    # Process elements sequentially in groups of 5
    for i in range(0, n - 5, 5):
        # Load values from read-only copy for reads, original for self-reads
        a_i = tl.load(a_ptr + i)
        a_i_plus_1 = tl.load(a_copy_ptr + i + 1)
        
        # First operation: a[i] = a[i + 1] * a[i]
        result_0 = a_i_plus_1 * a_i
        tl.store(a_ptr + i, result_0)
        
        # Second operation: a[i + 1] = a[i + 2] * a[i + 1]
        a_i_plus_2 = tl.load(a_copy_ptr + i + 2)
        result_1 = a_i_plus_2 * a_i_plus_1
        tl.store(a_ptr + i + 1, result_1)
        
        # Third operation: a[i + 2] = a[i + 3] * a[i + 2]
        a_i_plus_3 = tl.load(a_copy_ptr + i + 3)
        result_2 = a_i_plus_3 * a_i_plus_2
        tl.store(a_ptr + i + 2, result_2)
        
        # Fourth operation: a[i + 3] = a[i + 4] * a[i + 3]
        a_i_plus_4 = tl.load(a_copy_ptr + i + 4)
        result_3 = a_i_plus_4 * a_i_plus_3
        tl.store(a_ptr + i + 3, result_3)
        
        # Fifth operation: a[i + 4] = a[i + 5] * a[i + 4]
        a_i_plus_5 = tl.load(a_copy_ptr + i + 5)
        result_4 = a_i_plus_5 * a_i_plus_4
        tl.store(a_ptr + i + 4, result_4)

def s116_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Use single thread for sequential execution
    grid = (1,)
    
    s116_kernel[grid](
        a,
        a_copy,
        n
    )