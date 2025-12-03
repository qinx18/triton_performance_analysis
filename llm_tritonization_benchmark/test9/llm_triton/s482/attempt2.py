import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    
    # Check break condition for each element
    break_condition = c_vals > b_vals
    
    # Use prefix sum to find first break point
    # Elements should be processed only if no earlier element breaks
    break_cumsum = tl.cumsum(break_condition.to(tl.int32), axis=0)
    should_break = break_cumsum > 0
    
    # Also check if current element itself breaks
    current_breaks = break_condition
    
    # Process elements that don't break and haven't encountered a break yet
    process_mask = mask & ~should_break & ~current_breaks
    
    # Compute result
    result = tl.where(process_mask, a_vals + b_vals * c_vals, a_vals)
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s482_triton(a, b, c):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s482_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )