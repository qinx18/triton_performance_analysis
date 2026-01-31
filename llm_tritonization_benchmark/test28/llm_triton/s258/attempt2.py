import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid != 0:
        return
    
    s_val = 0.0
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        for i in range(BLOCK_SIZE):
            idx = block_start + i
            if idx >= n_elements:
                return
            
            if a_vals[i] > 0.0:
                s_val = d_vals[i] * d_vals[i]
            
            tl.store(s_expanded_ptr + idx, s_val)

@triton.jit
def s258_compute_kernel(b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr, s_expanded_ptr, 
                       n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    s_vals = tl.load(s_expanded_ptr + current_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
    aa_vals = tl.load(aa_ptr + current_offsets, mask=mask, other=0.0)
    
    b_vals = s_vals * c_vals + d_vals
    e_vals = (s_vals + 1.0) * aa_vals
    
    tl.store(b_ptr + current_offsets, b_vals, mask=mask)
    tl.store(e_ptr + current_offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e, len_2d):
    n = len_2d
    BLOCK_SIZE = 256
    
    s_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    grid_expand = (1,)
    s258_expand_s_kernel[grid_expand](
        a, d, s_expanded, n, BLOCK_SIZE
    )
    
    grid_compute = (triton.cdiv(n, BLOCK_SIZE),)
    s258_compute_kernel[grid_compute](
        b, c, d, e, aa[0], s_expanded, n, BLOCK_SIZE
    )