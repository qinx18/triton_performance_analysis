import triton
import triton.language as tl
import torch

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    e_vals = tl.load(e_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    condition = e_vals >= t
    
    # Load a and b values where condition is true
    a_vals = tl.load(a_ptr + idx, mask=mask & condition)
    b_vals = tl.load(b_ptr + idx, mask=mask & condition)
    d_vals = tl.load(d_ptr + idx, mask=mask & condition)
    
    # Compute updates
    a_update = a_vals + c_vals * d_vals
    b_update = b_vals + c_vals * c_vals
    
    # Store results where condition is true
    tl.store(a_ptr + idx, a_update, mask=mask & condition)
    tl.store(b_ptr + idx, b_update, mask=mask & condition)

def s272_triton(a, b, c, d, e, t):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s272_kernel[grid](
        a, b, c, d, e, t, n_elements, BLOCK_SIZE
    )