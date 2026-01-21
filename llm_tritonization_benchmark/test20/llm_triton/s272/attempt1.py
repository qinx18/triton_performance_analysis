import triton
import triton.language as tl
import torch

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Check condition
    condition = e_vals >= t
    
    # Load values where condition is true
    c_vals = tl.load(c_ptr + offsets, mask=mask & condition)
    d_vals = tl.load(d_ptr + offsets, mask=mask & condition)
    a_vals = tl.load(a_ptr + offsets, mask=mask & condition)
    b_vals = tl.load(b_ptr + offsets, mask=mask & condition)
    
    # Compute updates
    cd_product = c_vals * d_vals
    c_squared = c_vals * c_vals
    
    # Update arrays
    new_a_vals = a_vals + cd_product
    new_b_vals = b_vals + c_squared
    
    # Store results
    tl.store(a_ptr + offsets, new_a_vals, mask=mask & condition)
    tl.store(b_ptr + offsets, new_b_vals, mask=mask & condition)

def s272_triton(a, b, c, d, e, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s272_kernel[grid](a, b, c, d, e, t, n_elements, BLOCK_SIZE=BLOCK_SIZE)