import triton
import triton.language as tl
import torch

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    condition = e_vals >= t
    
    # Only load and compute for elements that meet the condition
    c_vals = tl.load(c_ptr + offsets, mask=mask & condition)
    d_vals = tl.load(d_ptr + offsets, mask=mask & condition)
    
    # Load current values of a and b
    a_vals = tl.load(a_ptr + offsets, mask=mask & condition)
    b_vals = tl.load(b_ptr + offsets, mask=mask & condition)
    
    # Compute updates
    a_new = a_vals + c_vals * d_vals
    b_new = b_vals + c_vals * c_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask & condition)
    tl.store(b_ptr + offsets, b_new, mask=mask & condition)

def s272_triton(a, b, c, d, e, t):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s272_kernel[grid](a, b, c, d, e, t, N, BLOCK_SIZE=BLOCK_SIZE)