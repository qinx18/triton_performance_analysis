import triton
import triton.language as tl
import torch

@triton.jit
def s272_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    e_vals = tl.load(e_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    
    # Conditional computation
    condition = e_vals >= t
    
    # Update a and b only where condition is true
    c_times_d = c_vals * d_vals
    c_squared = c_vals * c_vals
    
    new_a = tl.where(condition, a_vals + c_times_d, a_vals)
    new_b = tl.where(condition, b_vals + c_squared, b_vals)
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=mask)
    tl.store(b_ptr + idx, new_b, mask=mask)

def s272_triton(a, b, c, d, e, t):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s272_kernel[grid](
        a, b, c, d, e, 
        t, N, BLOCK_SIZE
    )