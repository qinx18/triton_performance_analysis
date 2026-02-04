import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < n_elements - 1)
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Check if a[i] >= 0 - if true, skip everything
    skip_all = a_vals >= 0.0
    
    # Check if b[i] >= 0 - if true, skip a update but do b update
    skip_a_update = b_vals >= 0.0
    
    # Update a[i] only if both conditions are false
    should_update_a = (~skip_all) & (~skip_a_update)
    a_new = tl.where(should_update_a, a_vals + c_vals * d_vals, a_vals)
    tl.store(a_ptr + offsets, a_new, mask=mask)
    
    # Update b[i+1] if we didn't skip everything
    should_update_b = ~skip_all
    b_new = c_vals + d_vals * e_vals
    store_offsets = offsets + 1
    store_mask = mask & should_update_b & (store_offsets < n_elements)
    tl.store(b_ptr + store_offsets, b_new, mask=store_mask)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE
    )