import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependencies
    # Use only one thread to process all elements sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process elements sequentially, 5 at a time as in original
    for base_idx in range(0, n - 5, 5):
        # Load the values we need for this group of 5 operations
        # Each operation reads from the copy and writes to original
        
        # a[i] = a[i + 1] * a[i]
        val0 = tl.load(a_copy_ptr + base_idx)
        val1 = tl.load(a_copy_ptr + base_idx + 1)
        result0 = val1 * val0
        tl.store(a_ptr + base_idx, result0)
        
        # a[i + 1] = a[i + 2] * a[i + 1]
        val2 = tl.load(a_copy_ptr + base_idx + 2)
        result1 = val2 * val1
        tl.store(a_ptr + base_idx + 1, result1)
        
        # a[i + 2] = a[i + 3] * a[i + 2]
        val3 = tl.load(a_copy_ptr + base_idx + 3)
        result2 = val3 * val2
        tl.store(a_ptr + base_idx + 2, result2)
        
        # a[i + 3] = a[i + 4] * a[i + 3]
        val4 = tl.load(a_copy_ptr + base_idx + 4)
        result3 = val4 * val3
        tl.store(a_ptr + base_idx + 3, result3)
        
        # a[i + 4] = a[i + 5] * a[i + 4]
        val5 = tl.load(a_copy_ptr + base_idx + 5)
        result4 = val5 * val4
        tl.store(a_ptr + base_idx + 4, result4)

def s116_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Use single thread since this must be sequential
    grid = (1,)
    BLOCK_SIZE = 256
    
    s116_kernel[grid](
        a, a_copy, n, BLOCK_SIZE=BLOCK_SIZE
    )