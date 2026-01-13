import torch
import triton
import triton.language as tl

@triton.jit
def s253_expand_kernel(a_ptr, b_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to handle scalar expansion
    if tl.program_id(0) != 0:
        return
    
    s_val = 0.0
    for i in range(n_elements):
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        d_val = tl.load(d_ptr + i)
        
        if a_val > b_val:
            s_val = a_val - b_val * d_val
        
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s253_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    s_vals = tl.load(s_expanded_ptr + block_start + offsets, mask=mask)
    
    # Compute condition
    cond = a_vals > b_vals
    
    # Compute s values (already expanded)
    new_s_vals = a_vals - b_vals * d_vals
    
    # Update c where condition is true
    new_c_vals = tl.where(cond, c_vals + s_vals, c_vals)
    
    # Update a where condition is true
    new_a_vals = tl.where(cond, s_vals, a_vals)
    
    # Store results
    tl.store(c_ptr + block_start + offsets, new_c_vals, mask=mask)
    tl.store(a_ptr + block_start + offsets, new_a_vals, mask=mask)

def s253_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create expanded scalar array
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar variable
    s253_expand_kernel[(1,)](
        a, b, d, s_expanded,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Parallel computation using expanded array
    s253_kernel[grid](
        a, b, c, d, s_expanded,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )