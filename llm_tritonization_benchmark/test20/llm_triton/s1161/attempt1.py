import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load values
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Create condition mask
    cond_mask = c_vals < 0.0
    
    # For condition true (c[i] < 0): b[i] = a[i] + d[i] * d[i]
    # Need to load a[i] for this case
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_result = a_vals + d_vals * d_vals
    
    # For condition false (c[i] >= 0): a[i] = c[i] + d[i] * e[i]
    a_result = c_vals + d_vals * e_vals
    
    # Store results based on condition
    # Store to b when condition is true
    tl.store(b_ptr + offsets, b_result, mask=mask & cond_mask)
    
    # Store to a when condition is false
    tl.store(a_ptr + offsets, a_result, mask=mask & ~cond_mask)

def s1161_triton(a, b, c, d, e):
    n = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )