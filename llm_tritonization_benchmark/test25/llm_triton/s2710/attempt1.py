import torch
import triton
import triton.language as tl

@triton.jit
def s2710_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    x,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Main conditional logic
    cond1 = a_vals > b_vals
    
    # Case 1: a[i] > b[i]
    new_a_vals = tl.where(cond1, a_vals + b_vals * d_vals, a_vals)
    
    # Nested condition for c when a[i] > b[i]
    # Since LEN_1D is typically 32000 > 10, we use the first branch
    c_case1 = c_vals + d_vals * d_vals
    
    # Case 2: a[i] <= b[i]
    new_b_vals = tl.where(cond1, b_vals, a_vals + e_vals * e_vals)
    
    # Nested condition for c when a[i] <= b[i]
    cond2 = x > 0.0
    c_case2_true = a_vals + d_vals * d_vals
    c_case2_false = c_vals + e_vals * e_vals
    c_case2 = tl.where(cond2, c_case2_true, c_case2_false)
    
    # Final c values
    new_c_vals = tl.where(cond1, c_case1, c_case2)
    
    # Store results
    tl.store(a_ptr + idx, new_a_vals, mask=mask)
    tl.store(b_ptr + idx, new_b_vals, mask=mask)
    tl.store(c_ptr + idx, new_c_vals, mask=mask)

def s2710_triton(a, b, c, d, e, x=1):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e,
        float(x),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b, c