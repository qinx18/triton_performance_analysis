import triton
import triton.language as tl
import torch

@triton.jit
def s274_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input arrays
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = c[i] + e[i] * d[i]
    a_vals = c_vals + e_vals * d_vals
    
    # Conditional logic: if (a[i] > 0.)
    condition = a_vals > 0.0
    
    # If condition is true: b[i] = a[i] + b[i]
    # If condition is false: a[i] = d[i] * e[i]
    b_vals = tl.where(condition, a_vals + b_vals, b_vals)
    a_vals = tl.where(condition, a_vals, d_vals * e_vals)
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s274_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s274_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )