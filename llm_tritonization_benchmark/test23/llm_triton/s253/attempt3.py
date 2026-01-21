import torch
import triton
import triton.language as tl

@triton.jit
def s253_expand_kernel(a_ptr, b_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single block processes all elements sequentially for scalar expansion
    if tl.program_id(0) != 0:
        return
    
    s_val = 0.0
    
    # Process all elements sequentially
    for elem_idx in range(n_elements):
        # Load individual elements
        a_val = tl.load(a_ptr + elem_idx)
        b_val = tl.load(b_ptr + elem_idx)
        d_val = tl.load(d_ptr + elem_idx)
        
        # Check condition and update s_val
        if a_val > b_val:
            s_val = a_val - b_val * d_val
            
        # Store expanded scalar value
        tl.store(s_expanded_ptr + elem_idx, s_val)

@triton.jit
def s253_compute_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load all required data
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
    s_vals = tl.load(s_expanded_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute condition
    cond = a_vals > b_vals
    
    # Update c where condition is true
    c_new = tl.where(cond, c_vals + s_vals, c_vals)
    
    # Update a where condition is true  
    a_new = tl.where(cond, s_vals, a_vals)
    
    # Store results
    tl.store(c_ptr + current_offsets, c_new, mask=mask)
    tl.store(a_ptr + current_offsets, a_new, mask=mask)

def s253_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar s using sequential processing
    grid_expand = (1,)
    s253_expand_kernel[grid_expand](
        a, b, d, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Parallel computation using expanded scalar
    grid_compute = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s253_compute_kernel[grid_compute](
        a, b, c, d, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )