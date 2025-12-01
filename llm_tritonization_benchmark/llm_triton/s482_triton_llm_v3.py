import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check if any c[i] > b[i] in this block
        break_condition = c_vals > b_vals
        if tl.any(break_condition & mask):
            # Find the first position where break occurs
            break_mask = break_condition & mask
            # Process only elements before the break
            valid_elements = current_offsets < n_elements
            
            # Create mask for elements to process (before break position)
            process_mask = valid_elements & ~break_mask
            
            # If break occurs at first element of block, don't process any
            first_break = tl.minimum(tl.where(break_mask, current_offsets, n_elements))
            process_mask = process_mask & (current_offsets < first_break)
            
            if tl.any(process_mask):
                a_vals = tl.load(a_ptr + current_offsets, mask=process_mask, other=0.0)
                result = a_vals + b_vals * c_vals
                tl.store(a_ptr + current_offsets, result, mask=process_mask)
            
            # Break out of the loop
            break
        else:
            # No break condition, process all valid elements in block
            a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
            result = a_vals + b_vals * c_vals
            tl.store(a_ptr + current_offsets, result, mask=mask)

def s482_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (1,)
    s482_kernel[grid](
        a, b, c, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )