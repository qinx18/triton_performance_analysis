import torch
import triton
import triton.language as tl

@triton.jit
def s323_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, i_val, BLOCK_SIZE: tl.constexpr):
    # Process single element at index i_val
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = i_val + offsets
    
    mask = offsets < 1  # Only process one element
    
    # Load b[i-1], c[i], d[i], e[i]
    b_prev = tl.load(b_ptr + idx - 1, mask=mask)
    c_val = tl.load(c_ptr + idx, mask=mask)
    d_val = tl.load(d_ptr + idx, mask=mask)
    e_val = tl.load(e_ptr + idx, mask=mask)
    
    # Compute a[i] = b[i-1] + c[i] * d[i]
    a_new = b_prev + c_val * d_val
    
    # Compute b[i] = a[i] + c[i] * e[i]
    b_new = a_new + c_val * e_val
    
    # Store results
    tl.store(a_ptr + idx, a_new, mask=mask)
    tl.store(b_ptr + idx, b_new, mask=mask)

def s323_triton(a, b, c, d, e):
    LEN_1D = a.shape[0]
    BLOCK_SIZE = 1
    
    # Process each element sequentially from i=1 to LEN_1D-1
    for i in range(1, LEN_1D):
        s323_kernel[(1,)](
            a, b, c, d, e, i, BLOCK_SIZE=BLOCK_SIZE
        )