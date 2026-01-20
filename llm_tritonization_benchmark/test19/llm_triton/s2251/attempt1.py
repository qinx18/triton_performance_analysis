import triton
import triton.language as tl
import torch

@triton.jit
def expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially
    if tl.program_id(0) == 0:
        s_val = 0.0
        for i in range(N):
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            tl.store(s_expanded_ptr + i, s_val)
            s_val = b_val + c_val

@triton.jit
def s2251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    a_vals = s_vals * e_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    b_vals = a_vals + d_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s2251_triton(a, b, c, d, e):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create temporary array for scalar expansion
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar s
    grid_expand = (1,)
    expand_s_kernel[grid_expand](b, c, s_expanded, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Parallel computation
    grid_compute = (triton.cdiv(N, BLOCK_SIZE),)
    s2251_kernel[grid_compute](a, b, c, d, e, s_expanded, N, BLOCK_SIZE=BLOCK_SIZE)