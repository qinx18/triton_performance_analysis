import torch
import triton
import triton.language as tl

@triton.jit
def s2251_expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n):
    """Compute expanded scalar array sequentially"""
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    s_val = 0.0
    for i in range(n):
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        s_val = b_val + c_val
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < n
    
    # Load s values (use 0.0 for first element)
    s_vals = tl.where(current_offsets == 0, 0.0, 
                     tl.load(s_expanded_ptr + current_offsets - 1, mask=mask, other=0.0))
    
    # Load other arrays
    e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute a[i] = s*e[i]
    a_vals = s_vals * e_vals
    
    # Store a[i]
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    
    # Compute b[i] = a[i]+d[i]
    b_vals = a_vals + d_vals
    
    # Store b[i]
    tl.store(b_ptr + current_offsets, b_vals, mask=mask)

def s2251_triton(a, b, c, d, e):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Phase 1: Compute expanded scalar array
    grid_expand = (1,)
    s2251_expand_s_kernel[grid_expand](
        b, c, s_expanded, n
    )
    
    # Phase 2: Compute main kernel in parallel
    grid_main = (triton.cdiv(n, BLOCK_SIZE),)
    s2251_kernel[grid_main](
        a, b, c, d, e, s_expanded, n, BLOCK_SIZE=BLOCK_SIZE
    )