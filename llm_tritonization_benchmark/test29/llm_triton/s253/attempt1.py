import torch
import triton
import triton.language as tl

@triton.jit
def s253_expand_kernel(a_ptr, b_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single program instance to handle scalar expansion sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    s_val = 0.0
    
    # Process all elements sequentially to maintain dependencies
    for i in range(n_elements):
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        if a_val > b_val:
            s_val = a_val - b_val * d_val
        
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s253_compute_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    
    # Apply condition
    cond = a_vals > b_vals
    
    # Update c where condition is true
    c_new = tl.where(cond, c_vals + s_vals, c_vals)
    
    # Update a where condition is true
    a_new = tl.where(cond, s_vals, a_vals)
    
    # Store results
    tl.store(c_ptr + offsets, c_new, mask=mask)
    tl.store(a_ptr + offsets, a_new, mask=mask)

def s253_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar with sequential processing
    grid = (1,)
    s253_expand_kernel[grid](a, b, d, s_expanded, n_elements, BLOCK_SIZE)
    
    # Phase 2: Apply updates in parallel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s253_compute_kernel[grid](a, b, c, d, s_expanded, n_elements, BLOCK_SIZE)
    
    return a, c