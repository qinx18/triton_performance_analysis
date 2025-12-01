import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the sequential logic with data dependencies
    # We need to handle the conditional increment of j based on c[i] > 0
    
    # Since j depends on previous iterations and conditional logic,
    # we need to process this sequentially per block
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    if block_start == 0:
        # Only the first block processes everything sequentially
        j = -1
        for i in range(n_elements):
            j += 1
            
            # Load values for current i
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            e_val = tl.load(e_ptr + i)
            
            # First assignment: a[j] = b[i] + d[i] * e[i]
            result1 = b_val + d_val * e_val
            tl.store(a_ptr + j, result1)
            
            # Conditional increment and assignment
            if c_val > 0.0:
                j += 1
                result2 = c_val + d_val * e_val
                tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2  # LEN_1D/2 from original code
    
    # Use a single block since we need sequential processing
    BLOCK_SIZE = 1024
    grid = (1,)  # Single block to maintain sequential order
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a