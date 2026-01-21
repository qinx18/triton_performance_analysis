import torch
import triton
import triton.language as tl

@triton.jit
def s2251_expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar s
    if tl.program_id(0) == 0:
        s_val = 0.0
        for i in range(N):
            s_val = tl.load(b_ptr + i) + tl.load(c_ptr + i)
            tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load s values from expanded array (shifted by 1 to get previous iteration values)
    s_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # For i=0, s should be 0.0, for i>0, use s_expanded[i-1]
    s_mask = (offsets > 0) & mask
    s_prev_offsets = offsets - 1
    s_prev_vals = tl.load(s_expanded_ptr + s_prev_offsets, mask=s_mask)
    s_vals = tl.where(s_mask, s_prev_vals, 0.0)
    
    # Load input arrays
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute a[i] = s*e[i]
    a_vals = s_vals * e_vals
    
    # Compute b[i] = a[i]+d[i]
    b_vals = a_vals + d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s2251_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded array for scalar s
    s_expanded = torch.zeros(N, dtype=torch.float32, device=a.device)
    
    # Phase 1: Expand scalar s
    grid1 = (1,)
    s2251_expand_s_kernel[grid1](b, c, s_expanded, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Compute main kernel in parallel
    grid2 = (triton.cdiv(N, BLOCK_SIZE),)
    s2251_kernel[grid2](a, b, c, d, e, s_expanded, N, BLOCK_SIZE=BLOCK_SIZE)