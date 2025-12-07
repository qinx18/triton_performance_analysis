import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    d_vals = tl.load(d_ptr + idx, mask=mask)
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    
    # Update a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    tl.store(a_ptr + idx, result, mask=mask)

def s481_triton(a, b, c, d):
    # Phase 1: Check for exit condition
    condition_mask = d < 0.0
    
    if torch.any(condition_mask):
        exit_idx = torch.argmax(condition_mask.int()).item()
        # Process only elements before exit point
        if exit_idx > 0:
            valid_a = a[:exit_idx]
            valid_b = b[:exit_idx]
            valid_c = c[:exit_idx]
            valid_d = d[:exit_idx]
            
            n_elements = exit_idx
            BLOCK_SIZE = 256
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            
            s481_kernel[grid](
                valid_a, valid_b, valid_c, valid_d,
                n_elements, BLOCK_SIZE
            )
    else:
        # No exit condition, process all elements
        n_elements = a.shape[0]
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        s481_kernel[grid](
            a, b, c, d,
            n_elements, BLOCK_SIZE
        )