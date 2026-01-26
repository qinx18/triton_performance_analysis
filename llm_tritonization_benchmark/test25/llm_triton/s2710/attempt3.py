import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Main conditional: if (a[i] > b[i])
    cond1 = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    # a[i] += b[i] * d[i]
    new_a_vals = tl.where(cond1, a_vals + b_vals * d_vals, a_vals)
    
    # Inner condition for branch 1: if (LEN_1D > 10)
    len_gt_10 = n_elements > 10
    # c[i] += d[i] * d[i] vs c[i] = d[i] * e[i] + 1.0
    branch1_c = tl.where(len_gt_10, c_vals + d_vals * d_vals, d_vals * e_vals + 1.0)
    
    # Branch 2: a[i] <= b[i] - need to use original a_vals for computation
    # b[i] = a[i] + e[i] * e[i]
    new_b_vals = tl.where(cond1, b_vals, a_vals + e_vals * e_vals)
    
    # Inner condition for branch 2: if (x > 0.)
    x_gt_0 = x > 0.0
    # c[i] = a[i] + d[i] * d[i] vs c[i] += e[i] * e[i] - use original a_vals
    branch2_c = tl.where(x_gt_0, a_vals + d_vals * d_vals, c_vals + e_vals * e_vals)
    
    # Final c values based on main condition
    final_c = tl.where(cond1, branch1_c, branch2_c)
    
    # Store results
    tl.store(a_ptr + idx, new_a_vals, mask=mask)
    tl.store(b_ptr + idx, new_b_vals, mask=mask)
    tl.store(c_ptr + idx, final_c, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e, float(x), n_elements, BLOCK_SIZE
    )