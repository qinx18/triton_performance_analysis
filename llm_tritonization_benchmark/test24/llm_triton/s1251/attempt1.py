import torch
import triton
import triton.language as tl

@triton.jit
def s1251_expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single block processes all elements sequentially to handle dependency
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    s_val = 0.0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
        
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                s_val = b_vals[i] + c_vals[i]
                tl.store(s_expanded_ptr + block_start + i, s_val)

@triton.jit
def s1251_compute_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask, other=0.0)
    
    # Compute updates
    new_b = a_vals + d_vals
    new_a = s_vals * e_vals
    
    # Store results
    tl.store(b_ptr + offsets, new_b, mask=mask)
    tl.store(a_ptr + offsets, new_a, mask=mask)

def s1251_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s1251_expand_s_kernel[grid](
        b, c, s_expanded, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Compute in parallel using expanded scalar
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1251_compute_kernel[grid](
        a, b, c, d, e, s_expanded, N, BLOCK_SIZE=BLOCK_SIZE
    )