import triton
import triton.language as tl
import torch

@triton.jit
def s1279_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Check nested conditions: a[i] < 0 AND b[i] > a[i]
    cond1 = a_vals < 0.0
    cond2 = b_vals > a_vals
    combined_cond = cond1 & cond2
    
    # Update c[i] = c[i] + d[i] * e[i] where conditions are met
    update = d_vals * e_vals
    c_new = tl.where(combined_cond, c_vals + update, c_vals)
    
    # Store result
    tl.store(c_ptr + idx, c_new, mask=mask)

def s1279_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )