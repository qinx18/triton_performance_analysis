import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, n_elements):
    pid = tl.program_id(axis=0)
    if pid == 0:
        s_val = 0.0
        for i in range(n_elements):
            a_val = tl.load(a_ptr + i)
            if a_val > 0.0:
                d_val = tl.load(d_ptr + i)
                s_val = d_val * d_val
            tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s258_compute_kernel(s_expanded_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    aa_vals = tl.load(aa_ptr + offsets, mask=mask, other=0.0)
    
    b_vals = s_vals * c_vals + d_vals
    e_vals = (s_vals + 1.0) * aa_vals
    
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    tl.store(e_ptr + offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e):
    n_elements = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array for expanded scalar s
    s_expanded = torch.zeros(n_elements, dtype=aa.dtype, device=aa.device)
    
    # Phase 1: Expand scalar s
    grid1 = (1,)
    s258_expand_s_kernel[grid1](a, d, s_expanded, n_elements)
    
    # Phase 2: Compute using expanded scalar array
    grid2 = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s258_compute_kernel[grid2](s_expanded, aa, b, c, d, e, n_elements, BLOCK_SIZE=BLOCK_SIZE)