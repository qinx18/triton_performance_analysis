import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, strip_start, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Load b[i] to check condition
    b_val = tl.load(b_ptr + idx)
    
    # Load other required values
    a_val = tl.load(a_ptr + idx)
    c_val = tl.load(c_ptr + idx)
    d_val = tl.load(d_ptr + idx)
    e_val = tl.load(e_ptr + idx)
    
    # Condition: b[i] < 0
    condition = b_val < 0.0
    
    # Branch 1: a[i] = c[i] + d[i] * e[i] (when b[i] >= 0)
    branch1_result = c_val + d_val * e_val
    
    # Branch 2: c[i+1] = a[i] + d[i] * d[i] (when b[i] < 0)
    branch2_result = a_val + d_val * d_val
    
    # Store results conditionally
    # For branch 1: store to a[i] when b[i] >= 0
    tl.store(a_ptr + idx, branch1_result, mask=~condition)
    
    # For branch 2: store to c[i+1] when b[i] < 0
    tl.store(c_ptr + idx + 1, branch2_result, mask=condition)

def s161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 1  # Must be 1 due to RAW dependency
    
    # Process strips sequentially due to c[i+1] dependency
    for strip_start in range(0, n_elements, BLOCK_SIZE):
        s161_kernel[(1,)](
            a, b, c, d, e,
            strip_start,
            BLOCK_SIZE=BLOCK_SIZE
        )