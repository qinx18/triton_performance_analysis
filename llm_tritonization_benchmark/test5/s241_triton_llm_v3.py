import torch
import triton
import triton.language as tl

@triton.jit
def s241_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load current values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        
        # First statement: a[i] = b[i] * c[i] * d[i]
        a_vals = b_vals * c_vals * d_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)
        
        # For second statement, we need a[i+1] values
        # Load a[i+1] with special handling for the last element
        next_offsets = current_offsets + 1
        next_mask = next_offsets < n_elements
        
        # Load a[i+1] values where valid
        a_next_vals = tl.load(a_ptr + next_offsets, mask=next_mask)
        
        # Second statement: b[i] = a[i] * a[i+1] * d[i]
        # Only compute where we have valid a[i+1] values
        compute_mask = mask & next_mask
        b_new_vals = tl.where(compute_mask, a_vals * a_next_vals * d_vals, b_vals)
        tl.store(b_ptr + current_offsets, b_new_vals, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    
    BLOCK_SIZE = 256
    
    s241_kernel[(1,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b