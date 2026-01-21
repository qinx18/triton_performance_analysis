import triton
import triton.language as tl

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load arrays
    a = tl.load(a_ptr + block_start + offsets, mask=mask)
    b = tl.load(b_ptr + block_start + offsets, mask=mask)
    c = tl.load(c_ptr + block_start + offsets, mask=mask)
    d = tl.load(d_ptr + block_start + offsets, mask=mask)
    e = tl.load(e_ptr + block_start + offsets, mask=mask)
    
    # Condition: a[i] > 0.0
    condition = a > 0.0
    
    # Compute both branches
    # Branch 1 (a[i] <= 0): b[i] = -b[i] + d[i] * e[i]
    b_new_branch1 = -b + d * e
    c_new_branch1 = c  # c unchanged in this branch
    
    # Branch 2 (a[i] > 0): c[i] = -c[i] + d[i] * e[i]  
    b_new_branch2 = b  # b unchanged in this branch
    c_new_branch2 = -c + d * e
    
    # Select based on condition
    b_new = tl.where(condition, b_new_branch2, b_new_branch1)
    c_new = tl.where(condition, c_new_branch2, c_new_branch1)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_new = b_new + c_new * d
    
    # Store results
    tl.store(a_ptr + block_start + offsets, a_new, mask=mask)
    tl.store(b_ptr + block_start + offsets, b_new, mask=mask)
    tl.store(c_ptr + block_start + offsets, c_new, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )