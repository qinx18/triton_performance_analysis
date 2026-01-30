import torch
import triton
import triton.language as tl

@triton.jit
def s2251_expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Expand scalar s using sequential computation"""
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    s_val = 0.0
    for i in range(n_elements):
        tl.store(s_expanded_ptr + i, s_val)
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        s_val = b_val + c_val

@triton.jit
def s2251_kernel(a_ptr, b_ptr, d_ptr, e_ptr, s_expanded_ptr, 
                 n_elements, BLOCK_SIZE: tl.constexpr):
    """Main computation kernel using expanded scalar array"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load expanded s values
    s_vals = tl.load(s_expanded_ptr + current_offsets, mask=mask)
    
    # Load other arrays
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    
    # Compute a[i] = s*e[i]
    a_vals = s_vals * e_vals
    
    # Store a values
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    
    # Compute b[i] = a[i]+d[i]
    b_vals = a_vals + d_vals
    
    # Store b values
    tl.store(b_ptr + current_offsets, b_vals, mask=mask)

def s2251_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(N, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s2251_expand_s_kernel[grid](
        b, c, s_expanded,
        N, BLOCK_SIZE
    )
    
    # Phase 2: Main computation
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s2251_kernel[grid](
        a, b, d, e, s_expanded,
        N, BLOCK_SIZE
    )
    
    return a