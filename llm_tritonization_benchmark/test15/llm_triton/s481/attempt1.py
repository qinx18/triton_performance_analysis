import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    result = a_vals + b_vals * c_vals
    tl.store(a_ptr + offsets, result, mask=mask)

def s481_triton(a, b, c, d):
    # Phase 1: Find exit condition
    condition_mask = d < 0.0
    
    if torch.any(condition_mask):
        exit_idx = torch.argmax(condition_mask.int()).item()
        valid_range = exit_idx
        
        if valid_range > 0:
            # Phase 2: Process only valid range
            BLOCK_SIZE = 256
            grid = (triton.cdiv(valid_range, BLOCK_SIZE),)
            
            s481_kernel[grid](a, b, c, valid_range, BLOCK_SIZE)
    else:
        # No exit condition, process all elements
        n_elements = a.shape[0]
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        s481_kernel[grid](a, b, c, n_elements, BLOCK_SIZE)