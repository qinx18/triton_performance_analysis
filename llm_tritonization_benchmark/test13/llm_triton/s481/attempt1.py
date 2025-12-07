import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    
    result = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + indices, result, mask=mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Check for early exit condition
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        exit_idx = torch.argmax(exit_mask.int()).item()
        valid_range = exit_idx
        if valid_range > 0:
            a_slice = a[:valid_range]
            b_slice = b[:valid_range]
            c_slice = c[:valid_range]
            
            BLOCK_SIZE = 256
            grid = (triton.cdiv(valid_range, BLOCK_SIZE),)
            
            s481_kernel[grid](
                a_slice, b_slice, c_slice,
                valid_range,
                BLOCK_SIZE=BLOCK_SIZE
            )
    else:
        # No exit condition triggered, process all elements
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        s481_kernel[grid](
            a, b, c,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )