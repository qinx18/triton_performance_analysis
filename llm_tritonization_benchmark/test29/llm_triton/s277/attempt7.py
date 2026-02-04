import torch
import triton
import triton.language as tl

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (n_elements - 1)
    
    # Load current values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute the values that would be written to b[i+1]
    new_b_vals = c_vals + d_vals * e_vals
    
    # Determine write conditions for b[i+1]
    # b[i+1] is written when a[i] < 0
    write_b_mask = a_vals < 0.0
    
    # Compute effective b values for i+1 positions
    # Load b[i+1] values
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < n_elements
    current_b_plus_1 = tl.load(b_ptr + offsets_plus_1, mask=mask_plus_1)
    
    # For each position i, determine if b[i] was written by previous iteration
    # This requires checking if the previous iteration (i-1) had a[i-1] < 0
    offsets_minus_1 = offsets - 1
    mask_minus_1 = offsets_minus_1 >= 0
    
    # Load values needed to check previous iteration condition
    a_prev = tl.load(a_ptr + offsets_minus_1, mask=mask_minus_1, other=1.0)
    c_prev = tl.load(c_ptr + offsets_minus_1, mask=mask_minus_1, other=0.0)
    d_prev = tl.load(d_ptr + offsets_minus_1, mask=mask_minus_1, other=0.0)
    e_prev = tl.load(e_ptr + offsets_minus_1, mask=mask_minus_1, other=0.0)
    
    # Check if previous iteration wrote to b[i]
    prev_wrote_b = (a_prev < 0.0) & mask_minus_1
    prev_b_value = c_prev + d_prev * e_prev
    
    # Effective b[i] value
    effective_b = tl.where(prev_wrote_b, prev_b_value, b_vals)
    
    # Now compute updates using effective values
    # Update a[i] only if both a[i] < 0 and effective_b[i] < 0
    update_a_mask = (a_vals < 0.0) & (effective_b < 0.0) & mask
    new_a_vals = a_vals + c_vals * d_vals
    final_a_vals = tl.where(update_a_mask, new_a_vals, a_vals)
    
    # Store updated a values
    tl.store(a_ptr + offsets, final_a_vals, mask=mask)
    
    # Store b[i+1] values where condition is met
    final_b_mask = write_b_mask & mask & mask_plus_1
    tl.store(b_ptr + offsets_plus_1, new_b_vals, mask=final_b_mask)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b