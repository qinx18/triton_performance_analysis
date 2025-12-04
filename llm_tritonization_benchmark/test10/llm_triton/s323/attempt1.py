import triton
import triton.language as tl
import torch

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the recurrence sequentially since each iteration depends on the previous
    for i in range(1, n_elements):
        # Load scalar values
        b_prev = tl.load(b_ptr + (i - 1))
        c_i = tl.load(c_ptr + i)
        d_i = tl.load(d_ptr + i)
        e_i = tl.load(e_ptr + i)
        
        # Compute a[i] = b[i-1] + c[i] * d[i]
        a_i = b_prev + c_i * d_i
        tl.store(a_ptr + i, a_i)
        
        # Compute b[i] = a[i] + c[i] * e[i]
        b_i = a_i + c_i * e_i
        tl.store(b_ptr + i, b_i)

def s323_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Launch single thread since this is a sequential recurrence
    BLOCK_SIZE = 1
    grid = (1,)
    
    s323_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )