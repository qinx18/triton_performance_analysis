import torch
import triton
import triton.language as tl

@triton.jit
def s2710_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements, x,
    BLOCK_SIZE: tl.constexpr,
    LEN_1D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask)
    
    # Main conditional logic: if (a[i] > b[i])
    cond1 = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    new_a_1 = a_vals + b_vals * d_vals
    
    # Nested condition: if (LEN_1D > 10)
    if LEN_1D > 10:
        new_c_1 = c_vals + d_vals * d_vals
    else:
        new_c_1 = d_vals * e_vals + 1.0
    
    # Branch 2: a[i] <= b[i]
    new_b_2 = a_vals + e_vals * e_vals
    
    # Nested condition: if (x > 0.)
    if x > 0.0:
        new_c_2 = a_vals + d_vals * d_vals
    else:
        new_c_2 = c_vals + e_vals * e_vals
    
    # Apply conditional updates
    final_a = tl.where(cond1, new_a_1, a_vals)
    final_b = tl.where(cond1, b_vals, new_b_2)
    final_c = tl.where(cond1, new_c_1, new_c_2)
    
    # Store results
    tl.store(a_ptr + block_start + offsets, final_a, mask=mask)
    tl.store(b_ptr + block_start + offsets, final_b, mask=mask)
    tl.store(c_ptr + block_start + offsets, final_c, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    LEN_1D = n_elements
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e,
        n_elements, float(x),
        BLOCK_SIZE=BLOCK_SIZE,
        LEN_1D=LEN_1D,
    )