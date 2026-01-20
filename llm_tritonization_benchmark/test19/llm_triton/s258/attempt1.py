import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Single thread processes all elements sequentially to maintain dependencies
    if pid == 0:
        s_val = 0.0
        for i in range(n_elements):
            a_val = tl.load(a_ptr + i)
            if a_val > 0.0:
                d_val = tl.load(d_ptr + i)
                s_val = d_val * d_val
            tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s258_compute_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, 
                        n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
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

def s258_triton(a, aa, b, c, d, e):
    n_elements = aa.shape[0] * aa.shape[1]  # LEN_2D equivalent
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Get flat view of 2D arrays
    aa_flat = aa.view(-1)
    
    # Phase 1: Expand scalar s
    grid = (1,)  # Single block for sequential processing
    s258_expand_s_kernel[grid](a, d, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Compute results in parallel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s258_compute_kernel[grid](a, aa_flat, b, c, d, e, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)