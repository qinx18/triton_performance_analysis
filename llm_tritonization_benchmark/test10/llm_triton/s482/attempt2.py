import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # Check break condition - if any c[i] > b[i] in current block
        break_mask = c_vals > b_vals
        
        # Find first position where break occurs
        break_positions = tl.where(break_mask, current_offsets, n_elements)
        min_break_pos = tl.min(break_positions)
        
        # Only compute for elements before break position
        compute_mask = mask & (current_offsets < min_break_pos)
        
        # Perform computation only for valid elements
        result = tl.where(compute_mask, a_vals + b_vals * c_vals, a_vals)
        
        # Store results
        tl.store(a_ptr + current_offsets, result, mask=mask)
        
        # Check if we need to stop processing further blocks
        stop_condition = min_break_pos < (block_start + BLOCK_SIZE)
        if stop_condition:
            return

def s482_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (1,)
    s482_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=BLOCK_SIZE)