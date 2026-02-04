import triton
import triton.language as tl
import torch

@triton.jit
def s279_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Condition: if (a[i] > 0.)
    cond1 = a_vals > 0.0
    
    # Path when a[i] <= 0.0
    # b[i] = -b[i] + d[i] * d[i]
    b_new_path1 = -b_vals + d_vals * d_vals
    
    # Condition: if (b[i] <= a[i])
    cond2 = b_new_path1 <= a_vals
    
    # Path when a[i] <= 0.0 and b[i] > a[i]
    # c[i] += d[i] * e[i]
    c_path1_branch2 = c_vals + d_vals * e_vals
    
    # Path when a[i] > 0.0 (L20)
    # c[i] = -c[i] + e[i] * e[i]
    c_path2 = -c_vals + e_vals * e_vals
    
    # Select final b values
    b_final = tl.where(cond1, b_vals, b_new_path1)
    
    # Select final c values
    # If a[i] > 0.0, use path2 (L20)
    # If a[i] <= 0.0 and b[i] <= a[i], keep original c
    # If a[i] <= 0.0 and b[i] > a[i], use path1_branch2
    c_path1 = tl.where(cond2, c_vals, c_path1_branch2)
    c_final = tl.where(cond1, c_path2, c_path1)
    
    # L30: a[i] = b[i] + c[i] * d[i]
    a_final = b_final + c_final * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_final, mask=mask)
    tl.store(b_ptr + offsets, b_final, mask=mask)
    tl.store(c_ptr + offsets, c_final, mask=mask)

def s279_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )