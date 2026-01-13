import torch
import triton
import triton.language as tl

@triton.jit
def expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, n):
    # Single thread processes all elements sequentially
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
def s258_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, 
                n, stride_aa, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n
    
    # Load expanded s values
    s_vals = tl.load(s_expanded_ptr + current_offsets, mask=mask)
    
    # Load arrays
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    aa_vals = tl.load(aa_ptr + current_offsets, mask=mask)
    
    # Compute b[i] = s * c[i] + d[i]
    b_vals = s_vals * c_vals + d_vals
    
    # Compute e[i] = (s + 1.) * aa[0][i]
    e_vals = (s_vals + 1.0) * aa_vals
    
    # Store results
    tl.store(b_ptr + current_offsets, b_vals, mask=mask)
    tl.store(e_ptr + current_offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e):
    n = aa.shape[0]
    
    # Create expanded s array
    s_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    expand_s_kernel[(1,)](a, d, s_expanded, n)
    
    # Phase 2: Compute in parallel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    stride_aa = aa.stride(0)
    
    s258_kernel[grid](a, aa[0], b, c, d, e, s_expanded, 
                      n, stride_aa, BLOCK_SIZE)