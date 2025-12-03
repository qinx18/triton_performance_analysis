import torch
import triton
import triton.language as tl

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask)
    
    # Compute condition: a[i] > 0
    condition = a_vals > 0.0
    
    # If condition is false: b[i] = -b[i] + d[i] * e[i]
    b_new_false = -b_vals + d_vals * e_vals
    
    # If condition is true: c[i] = -c[i] + d[i] * e[i]
    c_new_true = -c_vals + d_vals * e_vals
    
    # Select values based on condition
    b_final = tl.where(condition, b_vals, b_new_false)
    c_final = tl.where(condition, c_new_true, c_vals)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_final = b_final + c_final * d_vals
    
    # Store results
    tl.store(a_ptr + block_start + offsets, a_final, mask=mask)
    tl.store(b_ptr + block_start + offsets, b_final, mask=mask)
    tl.store(c_ptr + block_start + offsets, c_final, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE
    )