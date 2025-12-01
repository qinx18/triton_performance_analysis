import torch
import triton
import triton.language as tl

@triton.jit
def s2711_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = block_id * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Apply conditional update: if b[i] != 0.0, then a[i] += b[i] * c[i]
    non_zero_mask = b_vals != 0.0
    update_vals = b_vals * c_vals
    result = tl.where(non_zero_mask, a_vals + update_vals, a_vals)
    
    # Store results
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s2711_triton(a, b, c):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2711_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a