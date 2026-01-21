import triton
import triton.language as tl

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
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
    b_updated_branch1 = -b_vals + d_vals * e_vals
    c_updated_branch1 = c_vals  # c unchanged in this branch
    
    # Branch 2 (a[i] > 0): c[i] = -c[i] + d[i] * e[i]
    b_updated_branch2 = b_vals  # b unchanged in this branch
    c_updated_branch2 = -c_vals + d_vals * e_vals
    
    # Select based on condition
    b_final = tl.where(condition, b_updated_branch2, b_updated_branch1)
    c_final = tl.where(condition, c_updated_branch2, c_updated_branch1)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_final = b_final + c_final * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_final, mask=mask)
    tl.store(b_ptr + offsets, b_final, mask=mask)
    tl.store(c_ptr + offsets, c_final, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )