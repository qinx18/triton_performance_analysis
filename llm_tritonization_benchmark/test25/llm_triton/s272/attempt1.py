import triton
import triton.language as tl
import torch

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Check condition
    cond = e_vals >= t
    
    # Only process elements that meet the condition
    if tl.any(cond):
        a_vals = tl.load(a_ptr + indices, mask=mask)
        b_vals = tl.load(b_ptr + indices, mask=mask)
        c_vals = tl.load(c_ptr + indices, mask=mask)
        d_vals = tl.load(d_ptr + indices, mask=mask)
        
        # Apply updates where condition is true
        update_mask = mask & cond
        new_a = a_vals + c_vals * d_vals
        new_b = b_vals + c_vals * c_vals
        
        tl.store(a_ptr + indices, new_a, mask=update_mask)
        tl.store(b_ptr + indices, new_b, mask=update_mask)

def s272_triton(a, b, c, d, e, t):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s272_kernel[grid](
        a, b, c, d, e,
        t, N, BLOCK_SIZE
    )
    
    return a, b