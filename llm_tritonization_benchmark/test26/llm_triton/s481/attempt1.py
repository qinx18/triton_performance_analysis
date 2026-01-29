import triton
import triton.language as tl
import torch

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    mask = offsets < n_elements
    
    # Load values
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Check for early exit condition
    exit_mask = d_vals < 0.0
    
    # If any element would trigger exit, we need to handle it
    # But in Triton, we can't exit globally, so we just process valid elements
    valid_mask = mask & ~exit_mask
    
    # Compute a[i] += b[i] * c[i] only for valid elements
    result = a_vals + b_vals * c_vals
    
    # Store result only for valid elements
    tl.store(a_ptr + offsets, result, mask=valid_mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # First, check if any element would trigger early exit
    exit_condition = d < 0.0
    if torch.any(exit_condition):
        # Find first exit index
        exit_idx = torch.argmax(exit_condition.int()).item()
        # Process only elements before exit point
        if exit_idx > 0:
            a_slice = a[:exit_idx]
            b_slice = b[:exit_idx]
            c_slice = c[:exit_idx]
            
            BLOCK_SIZE = 256
            grid = (triton.cdiv(exit_idx, BLOCK_SIZE),)
            
            s481_kernel[grid](
                a_slice, b_slice, c_slice, d[:exit_idx],
                exit_idx, BLOCK_SIZE
            )
        # Note: In original C code, exit(0) would terminate the program
        # Here we just stop processing at the exit point
        return
    
    # No exit condition, process all elements
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements, BLOCK_SIZE
    )