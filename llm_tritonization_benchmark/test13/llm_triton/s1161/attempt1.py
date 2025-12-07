import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    
    # Conditional computation
    condition = c_vals < 0.0
    
    # Path 1: c[i] >= 0, compute a[i] = c[i] + d[i] * e[i]
    a_new = c_vals + d_vals * e_vals
    
    # Path 2: c[i] < 0, compute b[i] = a[i] + d[i] * d[i]
    b_new = a_vals + d_vals * d_vals
    
    # Store results based on condition
    # For condition == False (c[i] >= 0): update a[i]
    tl.store(a_ptr + idx, a_new, mask=mask & ~condition)
    
    # For condition == True (c[i] < 0): update b[i] 
    tl.store(b_ptr + idx, b_new, mask=mask & condition)

def s1161_triton(a, b, c, d, e):
    n_elements = c.shape[0] - 1  # LEN_1D-1 from original loop
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )