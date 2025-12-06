import torch
import triton
import triton.language as tl

@triton.jit
def s279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load all arrays
    a = tl.load(a_ptr + indices, mask=mask)
    b = tl.load(b_ptr + indices, mask=mask)
    c = tl.load(c_ptr + indices, mask=mask)
    d = tl.load(d_ptr + indices, mask=mask)
    e = tl.load(e_ptr + indices, mask=mask)
    
    # Compute d*d and e*e once
    d_squared = d * d
    e_squared = e * e
    
    # Branch 1: a[i] > 0
    cond1 = a > 0.0
    
    # For cond1 == True (goto L20):
    # c[i] = -c[i] + e[i] * e[i]
    c_branch1 = -c + e_squared
    
    # For cond1 == False:
    # b[i] = -b[i] + d[i] * d[i]
    b_updated = -b + d_squared
    
    # Second condition: b[i] <= a[i] (only for cond1 == False)
    cond2 = b_updated <= a
    
    # For cond1 == False and cond2 == False:
    # c[i] += d[i] * e[i]
    c_branch2 = c + d * e
    
    # For cond1 == False and cond2 == True:
    # c stays the same
    
    # Select final c value
    c_final = tl.where(cond1, c_branch1, tl.where(cond2, c, c_branch2))
    
    # Select final b value (only updated when cond1 == False)
    b_final = tl.where(cond1, b, b_updated)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_final = b_final + c_final * d
    
    # Store results
    tl.store(a_ptr + indices, a_final, mask=mask)
    tl.store(b_ptr + indices, b_final, mask=mask)
    tl.store(c_ptr + indices, c_final, mask=mask)

def s279_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )