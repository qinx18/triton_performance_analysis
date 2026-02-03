import torch
import triton
import triton.language as tl

@triton.jit
def s258_expand_s_kernel(a_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Only process with first program
    if pid != 0:
        return
    
    s_val = 0.0
    
    # Process all elements sequentially to maintain dependencies
    idx = 0
    while idx < n_elements:
        a_val = tl.load(a_ptr + idx)
        if a_val > 0.0:
            d_val = tl.load(d_ptr + idx)
            s_val = d_val * d_val
        tl.store(s_expanded_ptr + idx, s_val)
        idx += 1

@triton.jit
def s258_compute_kernel(a_ptr, aa_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr,
                       n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load expanded s values
    s_vals = tl.load(s_expanded_ptr + current_offsets, mask=mask)
    
    # Load other arrays
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    aa_vals = tl.load(aa_ptr + current_offsets, mask=mask)
    
    # Compute results
    b_vals = s_vals * c_vals + d_vals
    e_vals = (s_vals + 1.0) * aa_vals
    
    # Store results
    tl.store(b_ptr + current_offsets, b_vals, mask=mask)
    tl.store(e_ptr + current_offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e, len_2d):
    n_elements = len_2d
    BLOCK_SIZE = 256
    
    # Create expanded s array
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s258_expand_s_kernel[grid](
        a, d, s_expanded, n_elements, BLOCK_SIZE
    )
    
    # Phase 2: Compute arrays in parallel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s258_compute_kernel[grid](
        a, aa, b, c, d, e, s_expanded, n_elements, BLOCK_SIZE
    )