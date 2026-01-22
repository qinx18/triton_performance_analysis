import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single program processes all elements sequentially to handle dependency
    pid = tl.program_id(0)
    
    if pid != 0:
        return
    
    s_val = 0.0
    for i in range(n_elements):
        a_val = tl.load(a_ptr + i)
        if a_val > 0.0:
            d_val = tl.load(d_ptr + i)
            s_val = d_val * d_val
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s258_compute_kernel(s_expanded_ptr, c_ptr, d_ptr, b_ptr, e_ptr, aa_ptr, 
                       n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    s_vals = tl.load(s_expanded_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    aa_vals = tl.load(aa_ptr + indices, mask=mask)
    
    # Compute results
    b_vals = s_vals * c_vals + d_vals
    e_vals = (s_vals + 1.0) * aa_vals
    
    # Store results
    tl.store(b_ptr + indices, b_vals, mask=mask)
    tl.store(e_ptr + indices, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e):
    n_elements = aa.shape[0] * aa.shape[1]
    BLOCK_SIZE = 256
    
    # Flatten aa to get the first row
    aa_flat = aa.view(-1)
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s258_expand_s_kernel[grid](
        a, d, s_expanded, n_elements, BLOCK_SIZE
    )
    
    # Phase 2: Compute results in parallel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s258_compute_kernel[grid](
        s_expanded, c, d, b, e, aa_flat, 
        n_elements, BLOCK_SIZE
    )