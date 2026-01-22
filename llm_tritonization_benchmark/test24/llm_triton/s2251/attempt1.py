import torch
import triton
import triton.language as tl

@triton.jit
def expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Single thread processes all elements sequentially
    if pid == 0:
        s_val = 0.0
        for i in range(n_elements):
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            tl.store(s_expanded_ptr + i, s_val)
            s_val = b_val + c_val

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute a[i] = s * e[i]
    a_vals = s_vals * e_vals
    
    # Store a[i]
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Compute b[i] = a[i] + d[i]
    b_vals = a_vals + d_vals
    
    # Store b[i]
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid_expand = (1,)
    expand_s_kernel[grid_expand](b, c, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Compute in parallel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s2251_kernel[grid](a, b, c, d, e, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)