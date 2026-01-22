import torch
import triton
import triton.language as tl

@triton.jit
def expand_s_kernel(b_ptr, c_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid == 0:  # Only first thread processes sequentially
        s_val = 0.0
        for i in range(n_elements):
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            
            s_val = b_val + c_val * d_val
            tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s251_kernel(a_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    result = s_vals * s_vals
    tl.store(a_ptr + offsets, result, mask=mask)

def s251_triton(a, b, c, d):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    expand_s_kernel[grid](b, c, d, s_expanded, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Compute a[i] = s * s in parallel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s251_kernel[grid](a, s_expanded, N, BLOCK_SIZE=BLOCK_SIZE)