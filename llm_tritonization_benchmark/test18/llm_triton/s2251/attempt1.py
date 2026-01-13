import torch
import triton
import triton.language as tl

@triton.jit
def s2251_expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially
    if tl.program_id(0) == 0:
        s_val = 0.0
        for i in range(n_elements):
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            tl.store(s_expanded_ptr + i, s_val)
            s_val = b_val + c_val

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    current_offsets = block_start + offsets
    
    # Load expanded s values
    s_vals = tl.load(s_expanded_ptr + current_offsets, mask=mask)
    
    # Load other arrays
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    
    # Compute a[i] = s*e[i]
    a_vals = s_vals * e_vals
    
    # Compute b[i] = a[i]+d[i]
    b_vals = a_vals + d_vals
    
    # Store results
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    tl.store(b_ptr + current_offsets, b_vals, mask=mask)

def s2251_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s2251_expand_s_kernel[grid](b, c, s_expanded, N, BLOCK_SIZE)
    
    # Phase 2: Parallel computation
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    s2251_kernel[grid](a, b, c, d, e, s_expanded, N, BLOCK_SIZE)