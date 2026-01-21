import triton
import triton.language as tl
import torch

@triton.jit
def s116_kernel(a_ptr, a_copy_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must process strictly sequentially due to RAW dependencies
    # Use a single thread to handle all elements in order
    
    pid = tl.program_id(0)
    if pid > 0:  # Only process with the first block
        return
    
    # Process elements sequentially
    for i in range(0, n - 5, 5):
        # Process 5 consecutive elements as in original unrolled loop
        
        # a[i] = a[i + 1] * a[i]
        val_i = tl.load(a_copy_ptr + i)
        val_i_plus_1 = tl.load(a_copy_ptr + i + 1)
        result_i = val_i_plus_1 * val_i
        tl.store(a_ptr + i, result_i)
        
        # a[i + 1] = a[i + 2] * a[i + 1]
        val_i_plus_2 = tl.load(a_copy_ptr + i + 2)
        result_i_plus_1 = val_i_plus_2 * val_i_plus_1
        tl.store(a_ptr + i + 1, result_i_plus_1)
        
        # a[i + 2] = a[i + 3] * a[i + 2]
        val_i_plus_3 = tl.load(a_copy_ptr + i + 3)
        result_i_plus_2 = val_i_plus_3 * val_i_plus_2
        tl.store(a_ptr + i + 2, result_i_plus_2)
        
        # a[i + 3] = a[i + 4] * a[i + 3]
        val_i_plus_4 = tl.load(a_copy_ptr + i + 4)
        result_i_plus_3 = val_i_plus_4 * val_i_plus_3
        tl.store(a_ptr + i + 3, result_i_plus_3)
        
        # a[i + 4] = a[i + 5] * a[i + 4]
        val_i_plus_5 = tl.load(a_copy_ptr + i + 5)
        result_i_plus_4 = val_i_plus_5 * val_i_plus_4
        tl.store(a_ptr + i + 4, result_i_plus_4)

def s116_triton(a):
    n = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Use single block since processing must be sequential
    grid = (1,)
    BLOCK_SIZE = 256
    
    s116_kernel[grid](
        a,
        a_copy,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )