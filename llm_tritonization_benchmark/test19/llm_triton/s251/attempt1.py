import torch
import triton
import triton.language as tl

@triton.jit
def s251_expand_kernel(s_expanded_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread expands scalar s sequentially
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    s_val = 0.0
    for i in range(n_elements):
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        s_val = b_val + c_val * d_val
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s251_kernel(a_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    result = s_vals * s_vals
    tl.store(a_ptr + offsets, result, mask=mask)

def s251_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid1 = (1,)
    s251_expand_kernel[grid1](s_expanded, b, c, d, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Compute result in parallel
    grid2 = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s251_kernel[grid2](a, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)