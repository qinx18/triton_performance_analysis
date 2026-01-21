import torch
import triton
import triton.language as tl

@triton.jit
def s253_expand_kernel(a_ptr, b_ptr, d_ptr, s_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar s
    if tl.program_id(0) == 0:
        s_val = 0.0
        for i in range(n):
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            if a_val > b_val:
                s_val = a_val - b_val * d_val
            
            tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s253_kernel(a_ptr, b_ptr, c_ptr, d_ptr, s_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    
    # Apply condition
    cond = a_vals > b_vals
    
    # Update c where condition is true
    new_c = tl.where(cond, c_vals + s_vals, c_vals)
    tl.store(c_ptr + offsets, new_c, mask=mask)
    
    # Update a where condition is true
    new_a = tl.where(cond, s_vals, a_vals)
    tl.store(a_ptr + offsets, new_a, mask=mask)

def s253_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid1 = (1,)
    s253_expand_kernel[grid1](a, b, d, s_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Apply parallel computation
    grid2 = (triton.cdiv(n, BLOCK_SIZE),)
    s253_kernel[grid2](a, b, c, d, s_expanded, n, BLOCK_SIZE=BLOCK_SIZE)