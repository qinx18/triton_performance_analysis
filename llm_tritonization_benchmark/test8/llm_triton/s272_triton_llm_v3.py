import torch
import triton
import triton.language as tl

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Conditional computation
    cond_mask = e_vals >= t
    
    # Apply updates only where condition is met
    cd_product = c_vals * d_vals
    c_squared = c_vals * c_vals
    
    new_a = tl.where(cond_mask, a_vals + cd_product, a_vals)
    new_b = tl.where(cond_mask, b_vals + c_squared, b_vals)
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=mask)
    tl.store(b_ptr + idx, new_b, mask=mask)

def s272_triton(a, b, c, d, e, t):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s272_kernel[grid](
        a, b, c, d, e, t, n_elements, BLOCK_SIZE
    )