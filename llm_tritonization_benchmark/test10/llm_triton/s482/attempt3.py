import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_start = 0
    
    # Process one block at a time until break condition or end
    for _ in range(triton.cdiv(n_elements, BLOCK_SIZE)):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check break condition
        break_mask = mask & (c_vals > b_vals)
        
        # Find minimum position where break occurs
        break_positions = tl.where(break_mask, current_offsets, n_elements)
        min_break_pos = tl.min(break_positions)
        
        # Compute mask: only process elements before break position
        compute_mask = mask & (current_offsets < min_break_pos)
        
        # Perform computation
        result = tl.where(compute_mask, a_vals + b_vals * c_vals, a_vals)
        
        # Store results
        tl.store(a_ptr + current_offsets, result, mask=mask)
        
        # If break occurred in this block, we're done
        should_break = min_break_pos < n_elements
        
        # Update block_start for next iteration
        block_start += BLOCK_SIZE
        
        # Exit condition check - if break occurred or processed all elements
        if should_break or block_start >= n_elements:
            break

def s482_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (1,)
    s482_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)