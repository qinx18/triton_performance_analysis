import triton
import triton.language as tl
import torch

@triton.jit
def s279_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load all arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Condition: a[i] > 0
    cond1 = a_vals > 0.0
    
    # Path 1: a[i] <= 0
    # b[i] = -b[i] + d[i] * d[i]
    b_new_path1 = -b_vals + d_vals * d_vals
    
    # Check if b[i] <= a[i] for path1
    cond2 = b_new_path1 <= a_vals
    
    # Path 1a: b[i] > a[i]
    # c[i] += d[i] * e[i]
    c_new_path1a = c_vals + d_vals * e_vals
    
    # Path 1b: b[i] <= a[i]
    # c unchanged
    c_new_path1b = c_vals
    
    # Select c value for path 1
    c_new_path1 = tl.where(cond2, c_new_path1b, c_new_path1a)
    
    # Path 2: a[i] > 0 (L20)
    # c[i] = -c[i] + e[i] * e[i]
    c_new_path2 = -c_vals + e_vals * e_vals
    b_new_path2 = b_vals  # b unchanged in this path
    
    # Select final values based on first condition
    b_final = tl.where(cond1, b_new_path2, b_new_path1)
    c_final = tl.where(cond1, c_new_path2, c_new_path1)
    
    # L30: a[i] = b[i] + c[i] * d[i]
    a_final = b_final + c_final * d_vals
    
    # Store results
    tl.store(a_ptr + idx, a_final, mask=mask)
    tl.store(b_ptr + idx, b_final, mask=mask)
    tl.store(c_ptr + idx, c_final, mask=mask)

def s279_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )