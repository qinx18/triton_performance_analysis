import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s482_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Phase 1: Find exit index
    condition_mask = c > b
    
    if torch.any(condition_mask):
        # Find first index where c[i] > b[i]
        exit_idx = torch.argmax(condition_mask.int()).item()
        # Process elements 0..exit_idx (inclusive, since computation happens before break)
        valid_range = exit_idx + 1
        
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