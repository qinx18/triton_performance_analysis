import triton
import triton.language as tl
import torch

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Condition: a[i] > 0
    condition = a_vals > 0.0
    
    # Compute both branches
    # Branch 1 (a[i] <= 0): b[i] = -b[i] + d[i] * e[i]
    b_branch1 = -b_vals + d_vals * e_vals
    # c remains unchanged in branch 1
    c_branch1 = c_vals
    
    # Branch 2 (a[i] > 0): c[i] = -c[i] + d[i] * e[i]
    c_branch2 = -c_vals + d_vals * e_vals
    # b remains unchanged in branch 2
    b_branch2 = b_vals
    
    # Select based on condition
    final_b = tl.where(condition, b_branch2, b_branch1)
    final_c = tl.where(condition, c_branch2, c_branch1)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    final_a = final_b + final_c * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, final_a, mask=mask)
    tl.store(b_ptr + offsets, final_b, mask=mask)
    tl.store(c_ptr + offsets, final_c, mask=mask)

def s278_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s278_kernel[grid](a, b, c, d, e, N, BLOCK_SIZE=BLOCK_SIZE)