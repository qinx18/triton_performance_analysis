import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_kernel(a_ptr, d_ptr, s_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    s_val = 0.0
    for i in range(n):
        a_val = tl.load(a_ptr + i)
        if a_val > 0.0:
            d_val = tl.load(d_ptr + i)
            s_val = d_val * d_val
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s258_compute_kernel(s_expanded_ptr, c_ptr, d_ptr, aa_ptr, b_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    aa_vals = tl.load(aa_ptr + offsets, mask=mask)
    
    b_vals = s_vals * c_vals + d_vals
    e_vals = (s_vals + 1.0) * aa_vals
    
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    tl.store(e_ptr + offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e):
    n = aa.shape[0] * aa.shape[1]
    
    s_expanded = torch.zeros(n, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 256
    
    # Phase 1: Expand scalar s
    grid1 = (1,)
    s258_expand_kernel[grid1](a, d, s_expanded, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Compute results in parallel  
    aa_flat = aa.view(-1)
    grid2 = (triton.cdiv(n, BLOCK_SIZE),)
    s258_compute_kernel[grid2](s_expanded, c, d, aa_flat, b, e, n, BLOCK_SIZE=BLOCK_SIZE)