import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the loop sequentially since j depends on conditional increments
    # We'll process one iteration at a time to maintain the exact semantics
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process sequentially to maintain j dependency
    j = -1
    for i in range(n_elements):
        j += 1
        
        # Load elements for current i
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        result1 = b_val + d_val * e_val
        tl.store(a_ptr + j, result1)
        
        # Check condition c[i] > 0
        if c_val > 0.0:
            j += 1
            # Second assignment: a[j] = c[i] + d[i] * e[i]
            result2 = c_val + d_val * e_val
            tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2  # Loop goes to LEN_1D/2
    
    BLOCK_SIZE = 1  # Sequential processing required due to j dependency
    
    grid = (1,)  # Single block since we need sequential execution
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )