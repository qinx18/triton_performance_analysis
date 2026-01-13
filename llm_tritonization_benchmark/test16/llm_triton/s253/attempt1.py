import torch
import triton
import triton.language as tl

@triton.jit
def s253_expand_s_kernel(a_ptr, b_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single program processes all elements sequentially for scalar expansion
    pid = tl.program_id(axis=0)
    if pid != 0:
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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    s_vals = tl.load(s_expanded_ptr + indices, mask=mask)
    
    cond = a_vals > b_vals
    
    # Only update c and a where condition is true
    new_s = a_vals - b_vals * d_vals
    c_updated = tl.where(cond, c_vals + s_vals, c_vals)
    a_updated = tl.where(cond, s_vals, a_vals)
    
    tl.store(c_ptr + indices, c_updated, mask=mask)
    tl.store(a_ptr + indices, a_updated, mask=mask)

def s253_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s253_expand_s_kernel[grid](
        a, b, d, s_expanded, n_elements, BLOCK_SIZE
    )
    
    # Phase 2: Parallel computation using expanded scalar
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s253_kernel[grid](
        a, b, c, d, s_expanded, n_elements, BLOCK_SIZE
    )