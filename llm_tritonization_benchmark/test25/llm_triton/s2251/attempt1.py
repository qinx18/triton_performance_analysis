import torch
import triton
import triton.language as tl

@triton.jit
def s2251_expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, N):
    """Phase 1: Expand scalar s into array using sequential processing"""
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    s_val = 0.0
    for i in range(N):
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        tl.store(s_expanded_ptr + i, s_val)
        s_val = b_val + c_val

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Phase 2: Parallel computation using expanded s array"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    a_vals = s_vals * e_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    b_vals = a_vals + d_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s2251_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array for expanded scalar
    s_expanded = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s2251_expand_s_kernel[grid](b, c, s_expanded, N)
    
    # Phase 2: Parallel computation
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s2251_kernel[grid](a, b, c, d, e, s_expanded, N, BLOCK_SIZE)