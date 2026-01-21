import triton
import triton.language as tl
import torch

@triton.jit
def s2251_expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar s
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    s_val = 0.0  # Initial value
    for i in range(n_elements):
        # s = b[i] + c[i] (computed before use in next iteration)
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        s_val = b_val + c_val
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit 
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Load expanded s values (shifted by 1 position, with s[0] = 0.0)
    s_vals = tl.zeros_like(offsets, dtype=tl.float32)
    # For i > 0, use s_expanded[i-1]
    prev_mask = (offsets > 0) & mask
    prev_offsets = offsets - 1
    s_vals = tl.where(prev_mask, 
                      tl.load(s_expanded_ptr + prev_offsets, mask=prev_mask), 
                      0.0)
    
    # Compute a[i] = s * e[i]
    a_vals = s_vals * e_vals
    
    # Compute b[i] = a[i] + d[i] 
    b_vals = a_vals + d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array for expanded scalar s
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid_expand = (1,)
    s2251_expand_s_kernel[grid_expand](
        b, c, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Compute main kernel in parallel
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s2251_kernel[grid](
        a, b, c, d, e, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )