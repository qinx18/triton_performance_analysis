import torch
import triton
import triton.language as tl

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in blocks, handling early termination
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check termination condition for each element
        term_condition = c_vals > b_vals
        
        # Find first element where termination occurs
        term_mask = term_condition & mask
        if tl.sum(term_mask.to(tl.int32)) > 0:
            # Find the minimum index where termination occurs
            term_indices = tl.where(term_mask, current_offsets, n_elements)
            min_term_idx = tl.min(term_indices)
            
            # Only process elements before termination
            process_mask = mask & (current_offsets < min_term_idx)
            
            # Compute and store
            result = a_vals + b_vals * c_vals
            tl.store(a_ptr + current_offsets, result, mask=process_mask)
            return
        
        # No termination in this block, process all valid elements
        result = a_vals + b_vals * c_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s482_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (1,)  # Single program to handle sequential processing with early exit
    
    s482_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )