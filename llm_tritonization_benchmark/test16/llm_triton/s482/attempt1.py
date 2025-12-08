import triton
import triton.language as tl
import torch

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, valid_range, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = (offsets < n_elements) & (offsets < valid_range)
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    result = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s482_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Phase 1: Find global exit index
    condition_mask = c > b
    
    if torch.any(condition_mask):
        exit_idx = torch.argmax(condition_mask.int()).item()
        valid_range = exit_idx + 1
    else:
        valid_range = n_elements
    
    # Phase 2: Process only valid range
    BLOCK_SIZE = 256
    grid = (triton.cdiv(valid_range, BLOCK_SIZE),)
    
    s482_kernel[grid](
        a, b, c, valid_range, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )