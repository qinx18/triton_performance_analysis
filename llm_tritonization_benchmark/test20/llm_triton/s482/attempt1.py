import triton
import triton.language as tl
import torch

def s482_triton(a, b, c):
    N = a.shape[0]
    
    # Find first index where c[i] > b[i] (exit condition)
    condition_mask = c > b
    
    if torch.any(condition_mask):
        # Find the first index where condition is true
        exit_idx = torch.argmax(condition_mask.int()).item()
        # Process elements 0 to exit_idx (inclusive, since computation happens before break)
        valid_length = exit_idx + 1
        
        if valid_length > 0:
            BLOCK_SIZE = 256
            grid = (triton.cdiv(valid_length, BLOCK_SIZE),)
            s482_kernel[grid](a, b, c, valid_length, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # No early exit, process all elements
        BLOCK_SIZE = 256
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s482_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)