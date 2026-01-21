import triton
import triton.language as tl
import torch

@triton.jit
def s258_expand_kernel(a_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single block processes all elements sequentially to handle dependencies
    pid = tl.program_id(axis=0)
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
def s258_compute_kernel(b_ptr, c_ptr, d_ptr, e_ptr, aa_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    aa_vals = tl.load(aa_ptr + offsets, mask=mask)
    
    b_vals = s_vals * c_vals + d_vals
    e_vals = (s_vals + 1.0) * aa_vals
    
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    tl.store(e_ptr + offsets, e_vals, mask=mask)

def s258_triton(a, aa, b, c, d, e):
    n_elements = aa.shape[0]  # Use LEN_2D from aa dimensions
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n_elements, device=a.device, dtype=a.dtype)
    
    # Get first row of aa for aa[0][i] access
    aa_row0 = aa[0]
    
    # Phase 1: Expand scalar s
    grid_expand = (1,)
    s258_expand_kernel[grid_expand](
        a, d, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Compute results in parallel
    grid_compute = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s258_compute_kernel[grid_compute](
        b, c, d, e, aa_row0, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )