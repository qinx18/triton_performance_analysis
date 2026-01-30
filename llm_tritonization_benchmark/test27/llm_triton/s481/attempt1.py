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
    
    # Check for early exit condition
    exit_condition = d_vals < 0.0
    
    # If any element in this block would cause exit, we need to be careful
    # Only process elements that are before any exit condition
    valid_mask = mask & ~exit_condition
    
    # Load other arrays only for valid elements
    a_vals = tl.load(a_ptr + indices, mask=valid_mask)
    b_vals = tl.load(b_ptr + indices, mask=valid_mask)
    c_vals = tl.load(c_ptr + indices, mask=valid_mask)
    
    # Compute result
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=valid_mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # First, check if there's any early exit condition
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        # Find first exit index
        exit_idx = torch.argmax(exit_mask.int()).item()
        # Process only elements before exit point
        if exit_idx > 0:
            a[:exit_idx] += b[:exit_idx] * c[:exit_idx]
        # Exit early (in original code this would call exit(0))
        return
    
    # No early exit, process all elements with Triton
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )