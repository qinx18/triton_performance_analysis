import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    
    # Perform computation
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s482_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Phase 1: Find exit point
    condition_mask = c > b
    
    if torch.any(condition_mask):
        exit_idx = torch.argmax(condition_mask.int()).item()
        # Include exit_idx because computation happens before exit check
        valid_range = exit_idx + 1
        
        # Phase 2: Process only valid range
        if valid_range > 0:
            BLOCK_SIZE = 256
            grid = (triton.cdiv(valid_range, BLOCK_SIZE),)
            
            s482_kernel[grid](
                a, b, c,
                valid_range,
                BLOCK_SIZE=BLOCK_SIZE
            )
    else:
        # No exit condition triggered, process all elements
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        s482_kernel[grid](
            a, b, c,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )