import triton
import triton.language as tl
import torch

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel must be executed with only one block since we need to handle early exit
    pid = tl.program_id(axis=0)
    
    # Process elements sequentially to handle the break condition
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load values
        b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
        a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        
        # Check for early exit condition
        should_break = c_vals > b_vals
        
        # Find the first element where break condition is true
        break_mask = mask & should_break
        if tl.any(break_mask):
            # Find the minimum index where break occurs
            break_indices = tl.where(break_mask, offsets, n_elements)
            min_break_idx = tl.min(break_indices)
            
            # Only process elements up to the break point
            process_mask = mask & (offsets < min_break_idx)
            
            if tl.any(process_mask):
                a_new = a_vals + b_vals * c_vals
                tl.store(a_ptr + offsets, a_new, mask=process_mask)
            
            # Exit the loop
            break
        else:
            # No break condition, process all masked elements
            a_new = a_vals + b_vals * c_vals
            tl.store(a_ptr + offsets, a_new, mask=mask)

def s482_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Launch with only one block to handle sequential break logic
    grid = (1,)
    
    s482_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )