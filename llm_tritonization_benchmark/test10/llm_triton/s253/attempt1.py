import torch
import triton
import triton.language as tl

@triton.jit
def s253_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    
    # Condition: a[i] > b[i]
    condition = a_vals > b_vals
    
    # Compute s = a[i] - b[i] * d[i]
    s_vals = a_vals - b_vals * d_vals
    
    # Update c[i] += s (only where condition is true)
    c_new = tl.where(condition, c_vals + s_vals, c_vals)
    
    # Update a[i] = s (only where condition is true)
    a_new = tl.where(condition, s_vals, a_vals)
    
    # Store results
    tl.store(c_ptr + block_start + offsets, c_new, mask=mask)
    tl.store(a_ptr + block_start + offsets, a_new, mask=mask)

def s253_triton(a, b, c, d):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s253_kernel[grid](
        a, b, c, d, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )