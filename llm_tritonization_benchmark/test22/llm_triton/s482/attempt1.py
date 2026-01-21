import triton
import triton.language as tl
import torch

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, exit_idx, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    exit_mask = offsets <= exit_idx
    combined_mask = mask & exit_mask
    
    a_vals = tl.load(a_ptr + offsets, mask=combined_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=combined_mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=combined_mask, other=0.0)
    
    result = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + offsets, result, mask=combined_mask)

def s482_triton(a, b, c):
    n = a.shape[0]
    
    # Find first index where c[i] > b[i] (exit condition)
    condition_mask = c > b
    
    if torch.any(condition_mask):
        exit_idx = torch.argmax(condition_mask.int()).item()
    else:
        exit_idx = n - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s482_kernel[grid](
        a, b, c, 
        exit_idx, n,
        BLOCK_SIZE=BLOCK_SIZE
    )