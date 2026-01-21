import triton
import triton.language as tl
import torch

@triton.jit
def s258_expand_s_kernel(a, d, s_expanded, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    s_val = 0.0
    for i in range(N):
        a_val = tl.load(a + i)
        if a_val > 0.0:
            d_val = tl.load(d + i)
            s_val = d_val * d_val
        tl.store(s_expanded + i, s_val)

@triton.jit
def s258_compute_kernel(a, aa, b, c, d, e, s_expanded, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    s_vals = tl.load(s_expanded + offsets, mask=mask)
    c_vals = tl.load(c + offsets, mask=mask)
    d_vals = tl.load(d + offsets, mask=mask)
    aa_vals = tl.load(aa + offsets, mask=mask)
    
    b_vals = s_vals * c_vals + d_vals
    e_vals = (s_vals + 1.0) * aa_vals
    
    tl.store(b + offsets, b_vals, mask=mask)
    tl.store(e + offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    s_expanded = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s258_expand_s_kernel[grid](a, d, s_expanded, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Compute in parallel using expanded values
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s258_compute_kernel[grid](a, aa, b, c, d, e, s_expanded, N, BLOCK_SIZE=BLOCK_SIZE)