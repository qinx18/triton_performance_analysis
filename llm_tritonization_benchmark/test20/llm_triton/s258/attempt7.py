import torch
import triton
import triton.language as tl

@triton.jit
def s258_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Handle scalar expansion sequentially
    if pid == 0:
        s_val = 0.0
        for i in range(n_elements):
            a_val = tl.load(a_ptr + i)
            if a_val > 0.0:
                d_val = tl.load(d_ptr + i)
                s_val = d_val * d_val
            
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            aa_val = tl.load(aa_ptr + i)
            
            b_val = s_val * c_val + d_val
            e_val = (s_val + 1.0) * aa_val
            
            tl.store(b_ptr + i, b_val)
            tl.store(e_ptr + i, e_val)

def s258_triton(a, aa, b, c, d, e):
    n_elements = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s258_kernel[grid](a, aa, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE)