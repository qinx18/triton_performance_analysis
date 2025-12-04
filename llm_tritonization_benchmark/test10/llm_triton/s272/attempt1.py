import triton
import triton.language as tl
import torch

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, t, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load data
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Conditional check
    cond = e_vals >= t
    
    # Load c and d values for computation
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Load current a and b values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute updates
    c_times_d = c_vals * d_vals
    c_squared = c_vals * c_vals
    
    # Apply conditional updates
    new_a = tl.where(cond, a_vals + c_times_d, a_vals)
    new_b = tl.where(cond, b_vals + c_squared, b_vals)
    
    # Store results
    tl.store(a_ptr + indices, new_a, mask=mask)
    tl.store(b_ptr + indices, new_b, mask=mask)

def s272_triton(a, b, c, d, e, t):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s272_kernel[grid](
        a, b, c, d, e,
        n_elements, t,
        BLOCK_SIZE=BLOCK_SIZE
    )