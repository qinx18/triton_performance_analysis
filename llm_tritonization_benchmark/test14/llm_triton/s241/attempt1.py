import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    
    # Compute a[i] = b[i] * c[i] * d[i]
    a_vals = b_vals * c_vals * d_vals
    
    # Store a[i]
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    
    # For b[i] = a[i] * a[i+1] * d[i], we need a[i+1]
    # Load a values from copy (for a[i]) and with offset (for a[i+1])
    a_current = tl.load(a_copy_ptr + current_offsets, mask=mask)
    
    # For a[i+1], we need offset+1, with mask for bounds
    next_offsets = current_offsets + 1
    next_mask = next_offsets < (n_elements + 1)  # Allow reading one past for valid indices
    mask_valid = mask & (current_offsets < (n_elements - 1))  # Only process i < n_elements-1
    
    a_next = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
    
    # Compute b[i] = a[i] * a[i+1] * d[i] (using updated a[i] values)
    b_new = a_vals * a_next * d_vals
    
    # Store b[i] only for valid indices
    tl.store(b_ptr + current_offsets, b_new, mask=mask_valid)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )