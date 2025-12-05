import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Check break condition: c[i] > b[i]
    break_mask = c_vals > b_vals
    
    # Find first position where break occurs
    break_positions = tl.arange(0, BLOCK_SIZE)
    valid_break = break_mask & mask
    
    # If any break condition is met, find the minimum position
    min_break_pos = tl.minimum.reduce(tl.where(valid_break, break_positions, BLOCK_SIZE))
    
    # Create execution mask: only execute if position is before first break
    exec_mask = mask & (tl.arange(0, BLOCK_SIZE) < min_break_pos)
    
    # Perform computation only for valid positions
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=exec_mask)

def s482_triton(a, b, c):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s482_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a