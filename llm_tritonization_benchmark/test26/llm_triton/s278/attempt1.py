import triton
import triton.language as tl
import torch

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
    
    # Compute d[i] * e[i] once
    de_product = d_vals * e_vals
    
    # Conditional computation: if a[i] > 0 then c[i] = -c[i] + d[i] * e[i]
    # else b[i] = -b[i] + d[i] * e[i]
    condition = a_vals > 0.0
    
    # Update b when condition is false
    b_vals = tl.where(condition, b_vals, -b_vals + de_product)
    
    # Update c when condition is true
    c_vals = tl.where(condition, -c_vals + de_product, c_vals)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    
    # Store results
    tl.store(a_ptr + block_start + offsets, a_vals, mask=mask)
    tl.store(b_ptr + block_start + offsets, b_vals, mask=mask)
    tl.store(c_ptr + block_start + offsets, c_vals, mask=mask)

def s278_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e, N, BLOCK_SIZE
    )