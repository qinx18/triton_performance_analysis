import torch
import triton
import triton.language as tl

@triton.jit
def s278_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all arrays
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # Condition: a[i] > 0
    condition = a > 0.0
    
    # Execute both branches
    # Branch 1 (else): b[i] = -b[i] + d[i] * e[i]
    b_new = -b + d * e
    
    # Branch 2 (if): c[i] = -c[i] + d[i] * e[i]  
    c_new = -c + d * e
    
    # Select which branch to use for b and c
    b_result = tl.where(condition, b, b_new)
    c_result = tl.where(condition, c_new, c)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    a_result = b_result + c_result * d
    
    # Store results
    tl.store(a_ptr + offsets, a_result, mask=mask)
    tl.store(b_ptr + offsets, b_result, mask=mask)
    tl.store(c_ptr + offsets, c_result, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE,
    )
    
    return a, b, c