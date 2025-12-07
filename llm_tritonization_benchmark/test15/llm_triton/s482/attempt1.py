import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    result = a_vals + b_vals * c_vals
    tl.store(a_ptr + indices, result, mask=mask)

def s482_triton(a, b, c):
    n_elements = a.shape[0]
    
    # Find exit condition
    condition_mask = c > b
    
    if torch.any(condition_mask):
        exit_idx = torch.argmax(condition_mask.int()).item()
        valid_range = exit_idx + 1
        
        BLOCK_SIZE = 256
        grid = (triton.cdiv(valid_range, BLOCK_SIZE),)
        
        s482_kernel[grid](
            a, b, c, valid_range,
            BLOCK_SIZE=BLOCK_SIZE
        )
    else:
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        s482_kernel[grid](
            a, b, c, n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )