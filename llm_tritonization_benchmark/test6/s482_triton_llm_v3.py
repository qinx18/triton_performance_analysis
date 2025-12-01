import triton
import triton.language as tl
import torch

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load data
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Check if any c[i] > b[i] in this block
        break_condition = c_vals > b_vals
        any_break = tl.any(break_condition & mask)
        
        if any_break:
            # Find first position where break occurs
            break_positions = tl.where(break_condition & mask, current_offsets, n_elements)
            first_break = tl.min(break_positions)
            
            # Only process elements before the break
            process_mask = mask & (current_offsets < first_break)
            
            # Compute and store
            result = a_vals + b_vals * c_vals
            tl.store(a_ptr + current_offsets, result, mask=process_mask)
            return
        
        # No break in this block, process all valid elements
        result = a_vals + b_vals * c_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s482_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    s482_kernel[(1,)](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )