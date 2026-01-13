import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Check for early exit condition
        exit_condition = d_vals < 0.0
        if tl.any(exit_condition):
            return
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        result = a_vals + b_vals * c_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Phase 1: Check for early exit condition on CPU
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        exit_idx = torch.argmax(exit_mask.int()).item()
        # Process only elements before the exit point
        if exit_idx > 0:
            valid_a = a[:exit_idx]
            valid_b = b[:exit_idx]
            valid_c = c[:exit_idx]
            valid_a += valid_b * valid_c
        return
    
    # Phase 2: No early exit, process all elements with Triton
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )