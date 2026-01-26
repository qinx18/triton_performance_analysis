import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar s
    if tl.program_id(0) == 0:
        s_val = 0.0
        for i in range(n_elements):
            a_val = tl.load(a_ptr + i)
            if a_val > 0.0:
                d_val = tl.load(d_ptr + i)
                s_val = d_val * d_val
            tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s258_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, 
                n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load expanded s values
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    
    # Load other arrays
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    aa_vals = tl.load(aa_ptr + offsets, mask=mask)
    
    # Compute results
    b_vals = s_vals * c_vals + d_vals
    e_vals = (s_vals + 1.0) * aa_vals
    
    # Store results
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    tl.store(e_ptr + offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e, len_2d):
    n_elements = len_2d
    
    # Create expanded s array
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s258_expand_s_kernel[grid](
        a, d, s_expanded, n_elements, BLOCK_SIZE
    )
    
    # Phase 2: Compute results in parallel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s258_kernel[grid](
        a, aa, b, c, d, e, s_expanded, n_elements, BLOCK_SIZE
    )