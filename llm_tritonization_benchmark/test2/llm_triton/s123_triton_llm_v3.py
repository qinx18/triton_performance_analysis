import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes the entire computation sequentially
    # due to the data-dependent induction variable j
    
    # Only process if this is the first block
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    j = -1
    
    for i in range(n_elements):
        j += 1
        
        # Load elements for first computation
        b_val = tl.load(b_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        c_val = tl.load(c_ptr + i)
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        result1 = b_val + d_val * e_val
        tl.store(a_ptr + j, result1)
        
        # Check condition and potentially do second assignment
        if c_val > 0.0:
            j += 1
            result2 = c_val + d_val * e_val
            tl.store(a_ptr + j, result2)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    
    # Launch with only one block since computation is sequential
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a