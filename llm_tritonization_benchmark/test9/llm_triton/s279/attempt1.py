import torch
import triton
import triton.language as tl

@triton.jit
def s279_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load all arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Control flow logic
    cond1 = a_vals > 0.0
    
    # Path when a[i] <= 0
    b_new = -b_vals + d_vals * d_vals
    cond2 = b_new <= a_vals
    c_path1 = tl.where(cond2, c_vals, c_vals + d_vals * e_vals)
    
    # Path when a[i] > 0 (L20)
    c_path2 = -c_vals + e_vals * e_vals
    
    # Select final c values based on first condition
    c_final = tl.where(cond1, c_path2, c_path1)
    
    # Update b only when a[i] <= 0
    b_final = tl.where(cond1, b_vals, b_new)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_final = b_final + c_final * d_vals
    
    # Store results
    tl.store(a_ptr + indices, a_final, mask=mask)
    tl.store(b_ptr + indices, b_final, mask=mask)
    tl.store(c_ptr + indices, c_final, mask=mask)

def s279_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )