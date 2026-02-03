import triton
import triton.language as tl
import torch

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    
    result = a_vals + b_vals * c_vals
    tl.store(a_ptr + indices, result, mask=mask)

def s482_triton(a, b, c):
    # Find exit point where c[i] > b[i]
    condition_mask = c > b
    
    if torch.any(condition_mask):
        exit_idx = torch.argmax(condition_mask.int()).item()
        # Process elements up to and including the exit point
        valid_range = exit_idx + 1
        
        if valid_range > 0:
            BLOCK_SIZE = 256
            n_elements = valid_range
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            
            s482_kernel[grid](
                a, b, c, n_elements, BLOCK_SIZE
            )
    else:
        # No exit condition triggered, process all elements
        BLOCK_SIZE = 256
        n_elements = a.shape[0]
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        s482_kernel[grid](
            a, b, c, n_elements, BLOCK_SIZE
        )