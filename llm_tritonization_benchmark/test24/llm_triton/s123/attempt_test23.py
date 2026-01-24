import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the loop sequentially since j depends on conditional increments
    # Each program handles one iteration of the outer timing loop
    
    # Initialize j to -1 (will be incremented to 0 first)
    j = -1
    
    # Process each element sequentially
    for i in range(0, n):
        # First increment and store
        j += 1
        
        # Load values for position i
        b_val = tl.load(b_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        result1 = b_val + d_val * e_val
        tl.store(a_ptr + j, result1)
        
        # Check condition and conditionally increment j and store again
        if c_val > 0.0:
            j += 1
            result2 = c_val + d_val * e_val
            tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n = b.shape[0] // 2  # LEN_1D/2
    
    # Launch single program since this cannot be parallelized due to j dependency
    grid = (1,)
    BLOCK_SIZE = 256
    
    s123_kernel[grid](
        a, b, c, d, e,
        n,
        BLOCK_SIZE=BLOCK_SIZE
    )