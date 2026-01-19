import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependencies
    # Use single thread to process all elements sequentially
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Process elements sequentially in chunks of 5 (preserving original unrolling)
    for i in range(0, n - 5, 5):
        # Load values from read-only copy
        a_i = tl.load(a_copy_ptr + i)
        a_i1 = tl.load(a_copy_ptr + i + 1)
        a_i2 = tl.load(a_copy_ptr + i + 2)
        a_i3 = tl.load(a_copy_ptr + i + 3)
        a_i4 = tl.load(a_copy_ptr + i + 4)
        a_i5 = tl.load(a_copy_ptr + i + 5)
        
        # Compute new values (sequential dependencies)
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
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Use single thread since computation must be sequential
    grid = (1,)
    BLOCK_SIZE = 256
    
    s116_kernel[grid](
        a, a_copy, n, 
        BLOCK_SIZE=BLOCK_SIZE
    )