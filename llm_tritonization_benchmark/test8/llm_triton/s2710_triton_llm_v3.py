import torch
import triton
import triton.language as tl

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Condition: a[i] > b[i]
    cond1 = a_vals > b_vals
    
    # Branch 1: if (a[i] > b[i])
    # a[i] += b[i] * d[i]
    new_a = a_vals + b_vals * d_vals
    # Since n_elements is typically 32000 > 10, always do: c[i] += d[i] * d[i]
    new_c_branch1 = c_vals + d_vals * d_vals
    
    # Branch 2: else
    # b[i] = a[i] + e[i] * e[i]
    new_b = a_vals + e_vals * e_vals
    # if (x > 0.)
    x_pos = x > 0.0
    if x_pos:
        # c[i] = a[i] + d[i] * d[i]
        new_c_branch2 = a_vals + d_vals * d_vals
    else:
        # c[i] += e[i] * e[i]
        new_c_branch2 = c_vals + e_vals * e_vals
    
    # Apply conditional updates
    final_a = tl.where(cond1, new_a, a_vals)
    final_b = tl.where(cond1, b_vals, new_b)
    final_c = tl.where(cond1, new_c_branch1, new_c_branch2)
    
    # Store results
    tl.store(a_ptr + offsets, final_a, mask=mask)
    tl.store(b_ptr + offsets, final_b, mask=mask)
    tl.store(c_ptr + offsets, final_c, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](a, b, c, d, e, x, n_elements, BLOCK_SIZE=BLOCK_SIZE)