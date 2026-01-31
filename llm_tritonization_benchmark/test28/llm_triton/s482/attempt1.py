import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute update
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s482_triton(a, b, c):
    N = a.shape[0]
    
    # Find exit condition index
    condition_mask = c > b
    
    if torch.any(condition_mask):
        exit_idx = torch.argmax(condition_mask.int()).item()
        # Include exit_idx because computation happens before exit check
        valid_range = exit_idx + 1
        
        # Create views for the valid range
        a_valid = a[:valid_range]
        b_valid = b[:valid_range]
        c_valid = c[:valid_range]
        
        BLOCK_SIZE = 256
        grid = (triton.cdiv(valid_range, BLOCK_SIZE),)
        
        s482_kernel[grid](
            a_valid, b_valid, c_valid,
            valid_range, BLOCK_SIZE
        )
    else:
        # No exit condition triggered, process all elements
        BLOCK_SIZE = 256
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        
        s482_kernel[grid](
            a, b, c,
            N, BLOCK_SIZE
        )