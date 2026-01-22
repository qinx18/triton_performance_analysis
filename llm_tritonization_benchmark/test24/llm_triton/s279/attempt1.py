import triton
import triton.language as tl
import torch

@triton.jit
def s279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < N
    
    # Load all arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Condition: a[i] > 0
    cond1 = a_vals > 0.0
    
    # Path 1: a[i] <= 0
    # b[i] = -b[i] + d[i] * d[i]
    b_new_path1 = -b_vals + d_vals * d_vals
    
    # Check if b[i] <= a[i]
    cond2 = b_new_path1 <= a_vals
    
    # If b[i] > a[i], then c[i] += d[i] * e[i]
    c_path1_branch1 = c_vals  # b[i] <= a[i], c unchanged
    c_path1_branch2 = c_vals + d_vals * e_vals  # b[i] > a[i]
    
    c_new_path1 = tl.where(cond2, c_path1_branch1, c_path1_branch2)
    
    # Path 2: a[i] > 0 (goto L20)
    # c[i] = -c[i] + e[i] * e[i]
    c_new_path2 = -c_vals + e_vals * e_vals
    
    # Select between paths based on cond1
    b_new = tl.where(cond1, b_vals, b_new_path1)  # b only changes in path 1
    c_new = tl.where(cond1, c_new_path2, c_new_path1)
    
    # L30: a[i] = b[i] + c[i] * d[i] (always executed)
    a_new = b_new + c_new * d_vals
    
    # Store results
    tl.store(a_ptr + idx, a_new, mask=mask)
    tl.store(b_ptr + idx, b_new, mask=mask)
    tl.store(c_ptr + idx, c_new, mask=mask)

def s279_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s279_kernel[grid](
        a, b, c, d, e,
        N, BLOCK_SIZE=BLOCK_SIZE
    )