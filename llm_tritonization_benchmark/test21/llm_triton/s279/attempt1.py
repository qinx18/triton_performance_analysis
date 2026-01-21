import triton
import triton.language as tl
import torch

@triton.jit
def s279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Compute using conditional logic
    # if (a[i] > 0.) goto L20; else continue to next line
    cond1 = a_vals > 0.0
    
    # Path when a[i] <= 0: b[i] = -b[i] + d[i] * d[i]
    b_new_path1 = -b_vals + d_vals * d_vals
    
    # if (b[i] <= a[i]) goto L30; else continue
    cond2 = b_new_path1 <= a_vals
    
    # Path when a[i] <= 0 AND b[i] > a[i]: c[i] += d[i] * e[i]
    c_new_path1_case1 = c_vals  # when b[i] <= a[i]
    c_new_path1_case2 = c_vals + d_vals * e_vals  # when b[i] > a[i]
    c_new_path1 = tl.where(cond2, c_new_path1_case1, c_new_path1_case2)
    b_new_path1_final = b_new_path1
    
    # L20 path when a[i] > 0: c[i] = -c[i] + e[i] * e[i]
    c_new_path2 = -c_vals + e_vals * e_vals
    b_new_path2 = b_vals  # b unchanged in this path
    
    # Select between the two main paths
    b_final = tl.where(cond1, b_new_path2, b_new_path1_final)
    c_final = tl.where(cond1, c_new_path2, c_new_path1)
    
    # L30: a[i] = b[i] + c[i] * d[i]
    a_final = b_final + c_final * d_vals
    
    # Store results
    tl.store(a_ptr + indices, a_final, mask=mask)
    tl.store(b_ptr + indices, b_final, mask=mask)
    tl.store(c_ptr + indices, c_final, mask=mask)

def s279_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )