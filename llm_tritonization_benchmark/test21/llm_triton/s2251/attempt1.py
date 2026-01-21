import torch
import triton
import triton.language as tl

@triton.jit
def s2251_expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid != 0:
        return
    
    s_val = 0.0
    for i in range(n_elements):
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
    
    # Load s values (from previous iteration, so offset by 1)
    s_prev_offsets = tl.maximum(offsets - 1, 0)
    s_mask = offsets > 0
    s_vals = tl.where(s_mask, 
                      tl.load(s_expanded_ptr + s_prev_offsets, mask=mask),
                      0.0)
    
    # Load arrays
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute a[i] = s*e[i] 
    a_vals = s_vals * e_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Load current s values
    s_curr_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    
    # Compute b[i] = a[i]+d[i]
    b_vals = a_vals + d_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s2251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s2251_expand_s_kernel[grid](b, c, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Parallel computation
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s2251_kernel[grid](a, b, c, d, e, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)