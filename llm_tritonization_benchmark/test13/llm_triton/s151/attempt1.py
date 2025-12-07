import torch
import triton
import triton.language as tl

@triton.jit
def s151_kernel(
    a_ptr, b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < (n_elements - 1)
    
    # Load b[i]
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Load a[i+1] (since m=1)
    a_shift_offsets = offsets + 1
    a_shift_mask = a_shift_offsets < n_elements
    combined_mask = mask & a_shift_mask
    a_shift_vals = tl.load(a_ptr + a_shift_offsets, mask=combined_mask)
    
    # Compute a[i] = a[i+1] + b[i]
    result = a_shift_vals + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + offsets, result, mask=combined_mask)

def s151_triton(a, b):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s151_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )