import torch
import triton
import triton.language as tl

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
    
    # Condition for first branch: a[i] > 0
    cond1 = a_vals > 0.0
    
    # Path when a[i] <= 0
    b_new = -b_vals + d_vals * d_vals
    cond2 = b_new <= a_vals
    
    # When a[i] <= 0 and b_new > a[i]: c[i] += d[i] * e[i]
    c_path1 = tl.where(cond2, c_vals, c_vals + d_vals * e_vals)
    
    # Path when a[i] > 0 (L20): c[i] = -c[i] + e[i] * e[i]
    c_path2 = -c_vals + e_vals * e_vals
    
    # Select final c values based on first condition
    c_final = tl.where(cond1, c_path2, c_path1)
    
    # Update b for all paths (b is only modified when a[i] <= 0)
    b_final = tl.where(cond1, b_vals, b_new)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
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