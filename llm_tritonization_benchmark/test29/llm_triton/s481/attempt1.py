import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load values
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Check if any element in this block triggers exit condition
    exit_condition = d_vals < 0.0
    
    # Only proceed if no exit condition in this block
    # This is a simplified approach - full correctness requires two-phase processing
    proceed_mask = mask & ~exit_condition
    
    # Load and compute only for valid elements
    a_vals = tl.load(a_ptr + indices, mask=proceed_mask)
    b_vals = tl.load(b_ptr + indices, mask=proceed_mask)
    c_vals = tl.load(c_ptr + indices, mask=proceed_mask)
    
    # Perform computation: a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=proceed_mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Two-phase approach for correct early exit handling
    # Phase 1: Check for exit condition
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        exit_idx = torch.argmax(exit_mask.int()).item()
        # Process only elements before exit point
        if exit_idx > 0:
            a[:exit_idx] += b[:exit_idx] * c[:exit_idx]
        # Exit early - don't process remaining elements
        return
    
    # Phase 2: No exit condition, process all elements with Triton kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )