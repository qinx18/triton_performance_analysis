import torch
import triton
import triton.language as tl

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    for i in range(n_elements):
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        j = 2 * i
        first_val = b_val + d_val * e_val
        tl.store(a_ptr + j, first_val)
        
        if c_val > 0.0:
            j = j + 1
            second_val = c_val + d_val * e_val
            tl.store(a_ptr + j, second_val)

def s123_triton(a, b, c, d, e):
    n_elements = b.shape[0] // 2
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s123_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a