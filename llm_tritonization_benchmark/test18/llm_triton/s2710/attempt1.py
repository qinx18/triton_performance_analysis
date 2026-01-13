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
    
    # Main conditional logic
    cond1 = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    new_a_branch1 = a_vals + b_vals * d_vals
    
    # Nested condition: LEN_1D > 10 (n_elements > 10)
    cond_len = n_elements > 10
    c_branch1_case1 = c_vals + d_vals * d_vals  # LEN_1D > 10
    c_branch1_case2 = d_vals * e_vals + 1.0     # LEN_1D <= 10
    
    c_branch1 = tl.where(cond_len, c_branch1_case1, c_branch1_case2)
    
    # Branch 2: a[i] <= b[i]  
    new_b_branch2 = a_vals + e_vals * e_vals
    
    # Nested condition: x > 0
    cond_x = x > 0.0
    c_branch2_case1 = a_vals + d_vals * d_vals  # x > 0
    c_branch2_case2 = c_vals + e_vals * e_vals  # x <= 0
    
    c_branch2 = tl.where(cond_x, c_branch2_case1, c_branch2_case2)
    
    # Apply main condition
    final_a = tl.where(cond1, new_a_branch1, a_vals)
    final_b = tl.where(cond1, b_vals, new_b_branch2)  
    final_c = tl.where(cond1, c_branch1, c_branch2)
    
    # Store results
    tl.store(a_ptr + block_start + offsets, final_a, mask=mask)
    tl.store(b_ptr + block_start + offsets, final_b, mask=mask)
    tl.store(c_ptr + block_start + offsets, final_c, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e,
        float(x),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )