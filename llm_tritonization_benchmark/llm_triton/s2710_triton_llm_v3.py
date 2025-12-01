import torch
import triton
import triton.language as tl

@triton.jit
def s2710_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements, x,
    BLOCK_SIZE: tl.constexpr,
    LEN_1D: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask)
    
    # Main conditional logic
    cond1 = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    a_new1 = a_vals + b_vals * d_vals
    if LEN_1D > 10:
        c_new1 = c_vals + d_vals * d_vals
    else:
        c_new1 = d_vals * e_vals + 1.0
    b_new1 = b_vals
    
    # Branch 2: a[i] <= b[i]
    a_new2 = a_vals
    b_new2 = a_vals + e_vals * e_vals
    if x > 0.0:
        c_new2 = a_vals + d_vals * d_vals
    else:
        c_new2 = c_vals + e_vals * e_vals
    
    # Select results based on condition
    a_result = tl.where(cond1, a_new1, a_new2)
    b_result = tl.where(cond1, b_new1, b_new2)
    c_result = tl.where(cond1, c_new1, c_new2)
    
    # Store results
    tl.store(a_ptr + block_start + offsets, a_result, mask=mask)
    tl.store(b_ptr + block_start + offsets, b_result, mask=mask)
    tl.store(c_ptr + block_start + offsets, c_result, mask=mask)

def s2710_triton(a, b, c, d, e, x, LEN_1D=32000):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e,
        n_elements, x,
        BLOCK_SIZE=BLOCK_SIZE,
        LEN_1D=LEN_1D,
    )