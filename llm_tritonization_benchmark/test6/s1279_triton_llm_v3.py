import torch
import triton
import triton.language as tl

@triton.jit
def s1279_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = block_id * BLOCK_SIZE
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    
    # Check conditions: a[i] < 0 and b[i] > a[i]
    cond1 = a_vals < 0.0
    cond2 = b_vals > a_vals
    both_conditions = cond1 & cond2
    
    # Update c[i] += d[i] * e[i] where both conditions are true
    update_vals = d_vals * e_vals
    c_vals = tl.where(both_conditions, c_vals + update_vals, c_vals)
    
    # Store result
    tl.store(c_ptr + current_offsets, c_vals, mask=mask)

def s1279_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )