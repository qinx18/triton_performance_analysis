import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        s_val = 0.0
        for i in range(n_elements):
            a_val = tl.load(a_ptr + i)
            if a_val > 0.0:
                d_val = tl.load(d_ptr + i)
                s_val = d_val * d_val
            tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s258_compute_kernel(aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, 
                        n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
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
    n_elements = aa.shape[0] * aa.shape[1]
    BLOCK_SIZE = 256
    
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    aa_flat = aa[0, :].contiguous()
    a_flat = a[:n_elements].contiguous()
    d_flat = d[:n_elements].contiguous()
    c_flat = c[:n_elements].contiguous()
    b_flat = b[:n_elements].contiguous()
    e_flat = e[:n_elements].contiguous()
    
    grid = (1,)
    s258_expand_s_kernel[grid](a_flat, d_flat, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s258_compute_kernel[grid](aa_flat, b_flat, c_flat, d_flat, e_flat, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)