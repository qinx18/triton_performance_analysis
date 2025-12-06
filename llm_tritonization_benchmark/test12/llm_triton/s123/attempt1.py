import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel cannot be parallelized due to data dependencies in j
    # We need to process sequentially
    pid = tl.program_id(0)
    
    # Only process with the first program
    if pid != 0:
        return
    
    # Process sequentially since j depends on previous iterations
    j = -1
    
    for i in range(n):
        j += 1
        
        # Load values for current i
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        result1 = b_val + d_val * e_val
        tl.store(a_ptr + j, result1)
        
        # Conditional assignment
        if c_val > 0.0:
            j += 1
            result2 = c_val + d_val * e_val
            tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n = b.shape[0] // 2  # LEN_1D/2
    
    # Launch with single block since we need sequential processing
    grid = (1,)
    BLOCK_SIZE = 1024
    
    s123_kernel[grid](
        a, b, c, d, e,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a