import torch
import triton
import triton.language as tl

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, n_elements, BLOCK_SIZE: tl.constexpr, LEN_1D: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Condition: a[i] > b[i]
    cond1 = a_vals > b_vals
    
    # Branch 1: if (a[i] > b[i])
    # a[i] += b[i] * d[i]
    new_a_vals = tl.where(cond1, a_vals + b_vals * d_vals, a_vals)
    
    # Nested condition: if (LEN_1D > 10)
    if LEN_1D > 10:
        # c[i] += d[i] * d[i]
        branch1_c = c_vals + d_vals * d_vals
    else:
        # c[i] = d[i] * e[i] + 1.0
        branch1_c = d_vals * e_vals + 1.0
    
    # Branch 2: else (a[i] <= b[i])
    # b[i] = a[i] + e[i] * e[i]
    new_b_vals = tl.where(cond1, b_vals, a_vals + e_vals * e_vals)
    
    # Nested condition: if (x > 0.0)
    if x > 0.0:
        # c[i] = a[i] + d[i] * d[i]
        branch2_c = a_vals + d_vals * d_vals
    else:
        # c[i] += e[i] * e[i]
        branch2_c = c_vals + e_vals * e_vals
    
    # Select c values based on main condition
    new_c_vals = tl.where(cond1, branch1_c, branch2_c)
    
    # Store results
    tl.store(a_ptr + indices, new_a_vals, mask=mask)
    tl.store(b_ptr + indices, new_b_vals, mask=mask)
    tl.store(c_ptr + indices, new_c_vals, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    LEN_1D = n_elements
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e, x, n_elements,
        BLOCK_SIZE=BLOCK_SIZE, LEN_1D=LEN_1D
    )