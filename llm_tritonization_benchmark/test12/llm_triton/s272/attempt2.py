import triton
import triton.language as tl
import torch

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, t, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Condition: e[i] >= t
    cond = e_vals >= t
    
    # Only process where both mask and condition are true
    active_mask = mask & cond
    
    a_vals = tl.load(a_ptr + idx, mask=active_mask)
    c_vals = tl.load(c_ptr + idx, mask=active_mask)
    d_vals = tl.load(d_ptr + idx, mask=active_mask)
    b_vals = tl.load(b_ptr + idx, mask=active_mask)
    
    # Compute updates
    a_new = a_vals + c_vals * d_vals
    b_new = b_vals + c_vals * c_vals
    
    # Store results
    tl.store(a_ptr + idx, a_new, mask=active_mask)
    tl.store(b_ptr + idx, b_new, mask=active_mask)

def s272_triton(a, b, c, d, e, t):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s272_kernel[grid](
        a, b, c, d, e,
        n_elements, t,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b