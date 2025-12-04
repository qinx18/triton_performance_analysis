import torch
import triton
import triton.language as tl

@triton.jit
def s1161_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Condition: c[i] < 0
    condition = c_vals < 0.0
    
    # For c[i] >= 0: a[i] = c[i] + d[i] * e[i]
    new_a_vals = c_vals + d_vals * e_vals
    
    # For c[i] < 0: b[i] = a[i] + d[i] * d[i]
    b_vals = a_vals + d_vals * d_vals
    
    # Update a where condition is false (c[i] >= 0)
    final_a_vals = tl.where(condition, a_vals, new_a_vals)
    
    # Store results
    tl.store(a_ptr + offsets, final_a_vals, mask=mask)
    tl.store(b_ptr + offsets, b_vals, mask=mask & condition)

def s1161_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1161_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )