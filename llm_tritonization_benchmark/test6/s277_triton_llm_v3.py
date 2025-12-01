import triton
import triton.language as tl
import torch

@triton.jit
def s277_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for start_idx in range(0, n_elements, BLOCK_SIZE):
        current_offsets = start_idx + offsets
        mask = current_offsets < n_elements
        
        # Load current values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Load b[i+1] for next indices
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (n_elements + 1)
        b_next_vals = tl.load(b_ptr + next_offsets, mask=next_mask, other=0.0)
        
        # Condition checks
        a_ge_zero = a_vals >= 0.0
        b_ge_zero = b_vals >= 0.0
        
        # Compute updates based on control flow logic
        # If a[i] >= 0, skip to L20 (do nothing)
        # If b[i] >= 0, skip to L30 (only update b[i+1])
        # Otherwise, update both a[i] and b[i+1]
        
        # Update a[i]: only if a[i] < 0 AND b[i] < 0
        should_update_a = (~a_ge_zero) & (~b_ge_zero)
        new_a_vals = tl.where(should_update_a, a_vals + c_vals * d_vals, a_vals)
        
        # Update b[i+1]: if a[i] < 0 (regardless of b[i] value)
        should_update_b_next = ~a_ge_zero
        new_b_next_vals = tl.where(should_update_b_next, c_vals + d_vals * e_vals, b_next_vals)
        
        # Store results
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
        tl.store(b_ptr + next_offsets, new_b_next_vals, mask=next_mask)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )