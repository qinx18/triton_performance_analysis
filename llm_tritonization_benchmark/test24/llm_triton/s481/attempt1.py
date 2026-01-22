import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    d_vals = tl.load(d_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    result = a_vals + b_vals * c_vals
    tl.store(a_ptr + indices, result, mask=mask)

def s481_triton(a, b, c, d):
    # Phase 1: Check for early exit condition
    exit_mask = d < 0.0
    
    if torch.any(exit_mask):
        # Find first exit point
        exit_idx = torch.argmax(exit_mask.int()).item()
        
        # Phase 2: Process only valid range (before exit point)
        if exit_idx > 0:
            n_elements = exit_idx
            BLOCK_SIZE = 256
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            
            s481_kernel[grid](
                a, b, c, d,
                n_elements=n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
        # If exit_idx == 0, no elements should be processed
    else:
        # No early exit, process all elements
        n_elements = a.shape[0]
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        s481_kernel[grid](
            a, b, c, d,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )