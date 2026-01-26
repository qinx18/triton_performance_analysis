import triton
import triton.language as tl
import torch

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask)
    
    # Condition: a[i] > 0.0
    condition = a_vals > 0.0
    
    # Compute both branches
    de_product = d_vals * e_vals
    branch_false = -b_vals + de_product  # b[i] = -b[i] + d[i] * e[i]
    branch_true = -c_vals + de_product   # c[i] = -c[i] + d[i] * e[i]
    
    # Update b and c based on condition
    new_b = tl.where(condition, b_vals, branch_false)
    new_c = tl.where(condition, branch_true, c_vals)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    new_a = new_b + new_c * d_vals
    
    # Store results
    tl.store(a_ptr + block_start + offsets, new_a, mask=mask)
    tl.store(b_ptr + block_start + offsets, new_b, mask=mask)
    tl.store(c_ptr + block_start + offsets, new_c, mask=mask)

def s278_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e, N, BLOCK_SIZE=BLOCK_SIZE
    )