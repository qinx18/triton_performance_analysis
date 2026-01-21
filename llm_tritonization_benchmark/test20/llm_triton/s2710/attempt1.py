import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    x,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Main conditional logic
    condition1 = a_vals > b_vals
    
    # True branch: a[i] > b[i]
    a_new = tl.where(condition1, a_vals + b_vals * d_vals, a_vals)
    
    # Nested condition for c array (true branch)
    c_true_branch = tl.where(n_elements > 10, c_vals + d_vals * d_vals, d_vals * e_vals + 1.0)
    
    # False branch: a[i] <= b[i]
    b_new = tl.where(condition1, b_vals, a_vals + e_vals * e_vals)
    
    # Nested condition for c array (false branch)
    c_false_branch = tl.where(x > 0.0, a_vals + d_vals * d_vals, c_vals + e_vals * e_vals)
    
    # Final c values
    c_new = tl.where(condition1, c_true_branch, c_false_branch)
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)
    tl.store(c_ptr + offsets, c_new, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e,
        float(x),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )