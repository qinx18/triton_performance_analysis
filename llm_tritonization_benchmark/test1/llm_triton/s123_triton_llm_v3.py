import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel cannot be efficiently parallelized due to data dependencies
    # The output index j depends on the conditional branch, creating irregular memory access
    # We'll process sequentially in blocks but handle the irregular indexing
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    if block_start == 0:  # Only first block processes everything sequentially
        j = 0  # Start from 0 (j starts at -1, then immediately incremented)
        
        for i in range(n_elements):
            # First assignment: a[j] = b[i] + d[i] * e[i]
            b_val = tl.load(b_ptr + i)
            d_val = tl.load(d_ptr + i)
            e_val = tl.load(e_ptr + i)
            tl.store(a_ptr + j, b_val + d_val * e_val)
            j += 1
            
            # Conditional assignment
            c_val = tl.load(c_ptr + i)
            if c_val > 0.0:
                tl.store(a_ptr + j, c_val + d_val * e_val)
                j += 1

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    
    BLOCK_SIZE = 1024
    grid = (1,)  # Single block to maintain sequential processing
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )