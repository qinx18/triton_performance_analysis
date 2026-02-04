import torch
import triton
import triton.language as tl

@triton.jit
def s253_expand_s_kernel(a_ptr, b_ptr, d_ptr, s_expanded_ptr, n_elements):
    # Single thread processes all elements sequentially to expand scalar s
    s_val = 0.0
    for i in range(n_elements):
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        if a_val > b_val:
            d_val = tl.load(d_ptr + i)
            s_val = a_val - b_val * d_val
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s253_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    
    # Apply conditional updates
    condition = a_vals > b_vals
    
    # Update c where condition is true
    c_updated = tl.where(condition, c_vals + s_vals, c_vals)
    
    # Update a where condition is true
    a_updated = tl.where(condition, s_vals, a_vals)
    
    # Store results
    tl.store(c_ptr + offsets, c_updated, mask=mask)
    tl.store(a_ptr + offsets, a_updated, mask=mask)

def s253_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar s (sequential)
    grid = (1,)
    s253_expand_s_kernel[grid](a, b, d, s_expanded, n_elements)
    
    # Phase 2: Apply updates in parallel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s253_kernel[grid](a, b, c, d, s_expanded, n_elements, BLOCK_SIZE)