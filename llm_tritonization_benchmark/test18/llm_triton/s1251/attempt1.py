import torch
import triton
import triton.language as tl

@triton.jit
def expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid == 0:  # Single thread processes all elements sequentially
        s_val = 0.0
        for i in range(n_elements):
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            s_val = b_val + c_val
            tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s1251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    s_vals = tl.load(s_expanded_ptr + current_offsets, mask=mask)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    
    # Compute
    b_new = a_vals + d_vals
    a_new = s_vals * e_vals
    
    # Store results
    tl.store(b_ptr + current_offsets, b_new, mask=mask)
    tl.store(a_ptr + current_offsets, a_new, mask=mask)

def s1251_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar
    grid = (1,)
    expand_s_kernel[grid](b, c, s_expanded, N, BLOCK_SIZE)
    
    # Phase 2: Parallel computation
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1251_kernel[grid](a, b, c, d, e, s_expanded, N, BLOCK_SIZE)