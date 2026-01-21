import torch
import triton
import triton.language as tl

@triton.jit
def s253_expand_kernel(a_ptr, b_ptr, d_ptr, s_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially for scalar expansion
    pid = tl.program_id(0)
    if pid != 0:
        return
    
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
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    
    condition = a_vals > b_vals
    
    new_s = a_vals - b_vals * d_vals
    final_s = tl.where(condition, new_s, s_vals)
    
    new_c = c_vals + final_s
    c_result = tl.where(condition, new_c, c_vals)
    
    a_result = tl.where(condition, final_s, a_vals)
    
    tl.store(c_ptr + offsets, c_result, mask=mask)
    tl.store(a_ptr + offsets, a_result, mask=mask)

def s253_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar with sequential kernel
    grid_expand = (1,)
    s253_expand_kernel[grid_expand](a, b, d, s_expanded, n, BLOCK_SIZE)
    
    # Phase 2: Use expanded array in parallel
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s253_kernel[grid](a, b, c, d, s_expanded, n, BLOCK_SIZE)