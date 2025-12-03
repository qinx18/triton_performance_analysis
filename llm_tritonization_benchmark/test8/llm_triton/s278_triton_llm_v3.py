import torch
import triton
import triton.language as tl

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask)
    
    # Implement the conditional logic: if a[i] > 0, modify c, else modify b
    condition = a_vals > 0.0
    
    # Apply operations conditionally
    b_new = tl.where(condition, b_vals, -b_vals + d_vals * e_vals)
    c_new = tl.where(condition, -c_vals + d_vals * e_vals, c_vals)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_new = b_new + c_new * d_vals
    
    # Store results
    tl.store(a_ptr + block_start + offsets, a_new, mask=mask)
    tl.store(b_ptr + block_start + offsets, b_new, mask=mask)
    tl.store(c_ptr + block_start + offsets, c_new, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](a, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE)