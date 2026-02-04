import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_kernel(a_ptr, d_ptr, s_expanded_ptr, n):
    """Single-threaded kernel to expand scalar s with conditional updates"""
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
def s258_compute_kernel(s_expanded_ptr, c_ptr, d_ptr, aa_ptr, b_ptr, e_ptr, 
                       n, BLOCK_SIZE: tl.constexpr):
    """Parallel kernel to compute final results using expanded scalar array"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    mask = (block_start + offsets) < n
    
    s_vals = tl.load(s_expanded_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    aa_vals = tl.load(aa_ptr + block_start + offsets, mask=mask)
    
    b_vals = s_vals * c_vals + d_vals
    e_vals = (s_vals + 1.0) * aa_vals
    
    tl.store(b_ptr + block_start + offsets, b_vals, mask=mask)
    tl.store(e_ptr + block_start + offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e, len_2d):
    n = len_2d
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s with sequential processing
    grid = (1,)
    s258_expand_kernel[grid](a, d, s_expanded, n)
    
    # Phase 2: Parallel computation using expanded array
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    grid = (num_blocks,)
    s258_compute_kernel[grid](s_expanded, c, d, aa[0], b, e, n, BLOCK_SIZE)