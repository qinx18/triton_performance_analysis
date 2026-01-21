import triton
import triton.language as tl

@triton.jit
def s2710_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Main conditional: if (a[i] > b[i])
    cond_a_gt_b = a_vals > b_vals
    
    # Branch 1: a[i] > b[i]
    new_a = a_vals + b_vals * d_vals
    # Since n_elements is derived from array size, we know LEN_1D > 10 for typical TSVC
    new_c_branch1 = c_vals + d_vals * d_vals
    
    # Branch 2: a[i] <= b[i]
    new_b = a_vals + e_vals * e_vals
    # Sub-conditional: x > 0
    cond_x_gt_0 = x > 0.0
    new_c_branch2_true = a_vals + d_vals * d_vals
    new_c_branch2_false = c_vals + e_vals * e_vals
    
    # Combine sub-conditional results
    new_c_branch2 = tl.where(cond_x_gt_0, new_c_branch2_true, new_c_branch2_false)
    
    # Apply main conditional
    final_a = tl.where(cond_a_gt_b, new_a, a_vals)
    final_b = tl.where(cond_a_gt_b, b_vals, new_b)
    final_c = tl.where(cond_a_gt_b, new_c_branch1, new_c_branch2)
    
    # Store results
    tl.store(a_ptr + offsets, final_a, mask=mask)
    tl.store(b_ptr + offsets, final_b, mask=mask)
    tl.store(c_ptr + offsets, final_c, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s2710_kernel[grid](a, b, c, d, e, x, N, BLOCK_SIZE=BLOCK_SIZE)