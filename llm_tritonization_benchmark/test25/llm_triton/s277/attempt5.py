import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    thread_id = tl.program_id(0)
    
    if thread_id != 0:
        return
    
    for i in range(n_elements - 1):
        a_val = tl.load(a_ptr + i)
        
        if a_val >= 0.0:
            continue
            
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        if b_val < 0.0:
            a_new = a_val + c_val * d_val
            tl.store(a_ptr + i, a_new)
        
        b_next_val = c_val + d_val * e_val
        tl.store(b_ptr + i + 1, b_next_val)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE,
    )
    
    return a, b