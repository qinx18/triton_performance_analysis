import triton
import triton.language as tl
import torch

@triton.jit
def s221_kernel(a_ptr, b_ptr, c_ptr, d_ptr, i_val, BLOCK_SIZE: tl.constexpr):
    # Single element processing at index i_val
    a_val = tl.load(a_ptr + i_val)
    c_val = tl.load(c_ptr + i_val)
    d_val = tl.load(d_ptr + i_val)
    
    # a[i] += c[i] * d[i]
    a_new = a_val + c_val * d_val
    tl.store(a_ptr + i_val, a_new)
    
    # b[i] = b[i-1] + a[i] + d[i]
    b_prev = tl.load(b_ptr + i_val - 1)
    b_new = b_prev + a_new + d_val
    tl.store(b_ptr + i_val, b_new)

def s221_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Process sequentially from i=1 to LEN_1D-1 due to loop-carried dependency
    for i in range(1, n_elements):
        s221_kernel[(1,)](
            a, b, c, d,
            i,
            BLOCK_SIZE=1
        )