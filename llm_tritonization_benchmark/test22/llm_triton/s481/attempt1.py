import torch
import triton
import triton.language as tl

def s481_triton(a, b, c, d):
    N = a.shape[0]
    
    # Find first index where d[i] < 0.0 (exit condition)
    condition_mask = d < 0.0
    
    if torch.any(condition_mask):
        # Find the first index where exit condition is true
        exit_idx = torch.argmax(condition_mask.int()).item()
        # Process only elements before the exit point
        valid_range = exit_idx
        if valid_range > 0:
            BLOCK_SIZE = 256
            grid = (triton.cdiv(valid_range, BLOCK_SIZE),)
            s481_kernel[grid](a, b, c, valid_range, BLOCK_SIZE=BLOCK_SIZE)
    else:
        # No exit condition triggered, process all elements
        BLOCK_SIZE = 256
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s481_kernel[grid](a, b, c, N, BLOCK_SIZE=BLOCK_SIZE)

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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