import triton
import triton.language as tl

@triton.jit
def s1213_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel must be strictly sequential due to loop-carried dependency
    # b[i] depends on b[i-1] from previous iteration
    
    # Use single thread to process all elements sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Sequential processing from i=1 to n-2
    for i in range(1, n-1):
        # a[i] = b[i-1] + c[i]
        b_prev = tl.load(b_ptr + i - 1)
        c_val = tl.load(c_ptr + i)
        a_val = b_prev + c_val
        tl.store(a_ptr + i, a_val)
        
        # b[i] = a[i+1] * d[i]
        a_next = tl.load(a_ptr + i + 1)
        d_val = tl.load(d_ptr + i)
        b_val = a_next * d_val
        tl.store(b_ptr + i, b_val)

def s1213_triton(a, b, c, d):
    n = a.shape[0]
    
    # Create read-only copy of 'a' to handle WAR dependency
    a_copy = a.clone()
    
    # Update kernel to use the copy for reads
    @triton.jit
    def s1213_kernel_with_copy(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        if pid != 0:
            return
        
        for i in range(1, n-1):
            # a[i] = b[i-1] + c[i]
            b_prev = tl.load(b_ptr + i - 1)
            c_val = tl.load(c_ptr + i)
            a_val = b_prev + c_val
            tl.store(a_ptr + i, a_val)
            
            # b[i] = a[i+1] * d[i] (read from original a_copy)
            a_next = tl.load(a_copy_ptr + i + 1)
            d_val = tl.load(d_ptr + i)
            b_val = a_next * d_val
            tl.store(b_ptr + i, b_val)
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single block for sequential execution
    
    s1213_kernel_with_copy[grid](
        a, a_copy, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE
    )