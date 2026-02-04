import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load d values to check exit condition
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    
    # Check if any d[i] < 0
    exit_condition = d_vals < 0.0
    
    # If exit condition is met, don't process any elements
    # (In practice, this is a simplification - proper implementation would need cross-block coordination)
    process_mask = mask & ~exit_condition
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=process_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=process_mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=process_mask, other=0.0)
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=process_mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Two-phase approach for early exit pattern
    # Phase 1: Check for exit condition
    exit_condition = d < 0.0
    if torch.any(exit_condition):
        # Find first exit index
        exit_idx = torch.argmax(exit_condition.int()).item()
        # Process only elements before exit point
        if exit_idx > 0:
            a[:exit_idx] += b[:exit_idx] * c[:exit_idx]
        return
    
    # Phase 2: No exit condition, process all elements
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )