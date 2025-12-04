import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(1, n_elements):
        # Process one element at a time to maintain dependencies
        if i < n_elements:
            # Load current values
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            c_prev_val = tl.load(c_ptr + (i - 1))
            
            # t = a[i] + b[i]
            t = a_val + b_val
            
            # a[i] = t + c[i-1]
            a_new = t + c_prev_val
            tl.store(a_ptr + i, a_new)
            
            # t = c[i] * d[i]
            t = c_val * d_val
            
            # c[i] = t
            tl.store(c_ptr + i, t)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    s261_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )