import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load d values to check exit condition
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=1.0)
    
    # Check if any d[i] < 0 in this block
    exit_condition = d_vals < 0.0
    
    # If no exit condition in this block, process normally
    if not tl.any(exit_condition):
        a_vals = tl.load(a_ptr + offsets, mask=mask)
        b_vals = tl.load(b_ptr + offsets, mask=mask)
        c_vals = tl.load(c_ptr + offsets, mask=mask)
        result = a_vals + b_vals * c_vals
        tl.store(a_ptr + offsets, result, mask=mask)
    else:
        # Find first exit position in this block
        exit_mask = exit_condition
        # Process only elements before the first exit condition
        process_mask = mask & (~exit_mask)
        
        # For elements where exit condition is met, we need to handle them one by one
        for i in range(BLOCK_SIZE):
            if block_start + i >= n_elements:
                break
            offset = block_start + i
            d_val = tl.load(d_ptr + offset)
            if d_val < 0.0:
                # Exit encountered, stop processing
                break
            # Process this element
            a_val = tl.load(a_ptr + offset)
            b_val = tl.load(b_ptr + offset)
            c_val = tl.load(c_ptr + offset)
            result = a_val + b_val * c_val
            tl.store(a_ptr + offset, result)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # First, check if there's any early exit condition
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        # Find the first index where exit condition is true
        exit_idx = torch.argmax(exit_mask.int()).item()
        # Process only elements before the exit point
        if exit_idx > 0:
            a[:exit_idx] += b[:exit_idx] * c[:exit_idx]
    else:
        # No exit condition, process all elements normally
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        s481_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE=BLOCK_SIZE)