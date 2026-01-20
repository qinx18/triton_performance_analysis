import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    exit_condition = d_vals < 0.0
    
    # If any element in this block has exit condition, we need to handle carefully
    # But since we need global coordination, we'll let the wrapper handle exit detection
    
    # For elements that should be processed (no exit condition)
    process_mask = mask & (exit_condition == 0)
    
    a_vals = tl.load(a_ptr + offsets, mask=process_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=process_mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=process_mask, other=0.0)
    
    result = a_vals + b_vals * c_vals
    tl.store(a_ptr + offsets, result, mask=process_mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Phase 1: Check for early exit condition
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        # Find first index where exit condition is true
        exit_idx = torch.argmax(exit_mask.int()).item()
        # Process only elements before the exit point
        if exit_idx > 0:
            a_slice = a[:exit_idx]
            b_slice = b[:exit_idx]
            c_slice = c[:exit_idx]
            a_slice += b_slice * c_slice
        # Exit early - don't process remaining elements
        return
    
    # Phase 2: No exit condition, process all elements with Triton
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )