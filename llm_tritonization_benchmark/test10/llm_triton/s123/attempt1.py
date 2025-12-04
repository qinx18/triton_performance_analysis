import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel processes one iteration at a time due to complex dependencies
    pid = tl.program_id(0)
    
    if pid != 0:  # Only run on first program
        return
    
    # Sequential processing due to irregular data dependencies
    for i in range(n_elements):
        # Load scalar values
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # First assignment: a[j] = b[i] + d[i] * e[i]
        first_val = b_val + d_val * e_val
        j_pos = 2 * i  # j starts at -1, increments to 0, 1, 2, 3...
        tl.store(a_ptr + j_pos, first_val)
        
        # Conditional assignment
        if c_val > 0.0:
            second_val = c_val + d_val * e_val
            tl.store(a_ptr + j_pos + 1, second_val)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    BLOCK_SIZE = 256
    
    # Launch single thread block to handle sequential dependencies
    grid = (1,)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a