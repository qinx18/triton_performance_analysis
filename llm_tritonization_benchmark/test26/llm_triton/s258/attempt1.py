import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, N):
    # Single thread processes all elements sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    s_val = 0.0
    for i in range(N):
        a_val = tl.load(a_ptr + i)
        if a_val > 0.0:
            d_val = tl.load(d_ptr + i)
            s_val = d_val * d_val
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s258_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load values
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    aa_vals = tl.load(aa_ptr + offsets, mask=mask)
    
    # Compute results
    b_vals = s_vals * c_vals + d_vals
    e_vals = (s_vals + 1.0) * aa_vals
    
    # Store results
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    tl.store(e_ptr + offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e):
    N = aa.shape[0]
    
    # Create expanded scalar array
    s_expanded = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s258_expand_s_kernel[grid](a, d, s_expanded, N)
    
    # Phase 2: Compute results in parallel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s258_kernel[grid](a, aa, b, c, d, e, s_expanded, N, BLOCK_SIZE)