import torch
import triton
import triton.language as tl

@triton.jit
def s222_kernel(a_ptr, b_ptr, c_ptr, e_ptr, start_idx, LEN_1D, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = start_idx + offsets
    
    mask = (idx >= 1) & (idx < LEN_1D)
    
    # Load data
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    e_prev_vals = tl.load(e_ptr + idx - 1, mask=mask)
    
    # Compute operations
    bc_product = b_vals * c_vals
    a_updated = a_vals + bc_product
    e_updated = e_prev_vals * e_prev_vals
    a_final = a_updated - bc_product
    
    # Store results
    tl.store(a_ptr + idx, a_final, mask=mask)
    tl.store(e_ptr + idx, e_updated, mask=mask)

def s222_triton(a, b, c, e):
    LEN_1D = a.shape[0]
    BLOCK_SIZE = 1  # Must be 1 due to loop-carried dependency
    
    # Process sequentially in strips of size 1
    for start_idx in range(1, LEN_1D, BLOCK_SIZE):
        s222_kernel[(1,)](
            a, b, c, e,
            start_idx,
            LEN_1D,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return a, e