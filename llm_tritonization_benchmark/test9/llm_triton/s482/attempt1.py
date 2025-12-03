import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    
    # Check break condition: if c[i] > b[i]
    break_condition = c_vals > b_vals
    
    # Find first position where break condition is true
    break_mask = tl.cumsum(break_condition.to(tl.int32), axis=0) == 0
    
    # Only process elements before the break point
    process_mask = mask & break_mask
    
    a_vals = tl.load(a_ptr + current_offsets, mask=process_mask)
    result = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + current_offsets, result, mask=process_mask)

def s482_triton(a, b, c):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s482_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )