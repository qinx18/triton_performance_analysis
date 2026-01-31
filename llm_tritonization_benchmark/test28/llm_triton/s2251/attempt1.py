import torch
import triton
import triton.language as tl

@triton.jit
def s2251_expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single program processes all elements sequentially to expand scalar s
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    s_val = 0.0
    for i in range(n_elements):
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        tl.store(s_expanded_ptr + i, s_val)
        s_val = b_val + c_val

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load expanded scalar values
    s_vals = tl.load(s_expanded_ptr + indices, mask=mask)
    
    # Load arrays
    e_vals = tl.load(e_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Compute a[i] = s*e[i]
    a_vals = s_vals * e_vals
    
    # Store a values
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # Compute b[i] = a[i]+d[i]
    b_new_vals = a_vals + d_vals
    
    # Store updated b values
    tl.store(b_ptr + indices, b_new_vals, mask=mask)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s2251_expand_s_kernel[grid](
        b, c, s_expanded,
        n_elements, BLOCK_SIZE
    )
    
    # Phase 2: Parallel computation using expanded scalar
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s2251_kernel[grid](
        a, b, c, d, e, s_expanded,
        n_elements, BLOCK_SIZE
    )