import torch
import triton
import triton.language as tl

@triton.jit
def s277_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get strip ID and calculate strip start
    strip_id = tl.program_id(0)
    strip_start = strip_id
    
    # Define offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = strip_start + offsets
    
    # Mask for valid elements
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Control flow logic
    # if a[i] >= 0.0 -> goto L20 (skip everything)
    a_skip = a_vals >= 0.0
    
    # if b[i] >= 0.0 -> goto L30 (skip a[i] update)
    b_skip = b_vals >= 0.0
    
    # Update a[i] only if both conditions are false
    update_a = ~a_skip & ~b_skip
    new_a = tl.where(update_a, a_vals + c_vals * d_vals, a_vals)
    
    # Update b[i+1] if a[i] < 0.0 (not skipping to L20)
    update_b = ~a_skip
    b_next_val = c_vals + d_vals * e_vals
    
    # Store updated a values
    tl.store(a_ptr + idx, new_a, mask=mask)
    
    # Store b[i+1] values with appropriate mask
    b_mask = mask & update_b & (idx + 1 < n_elements + 1)
    tl.store(b_ptr + idx + 1, b_next_val, mask=b_mask)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    # Must use strip size of 1 due to RAW dependency
    BLOCK_SIZE = 1
    num_strips = n_elements
    
    # Process strips sequentially
    for strip_id in range(num_strips):
        s277_kernel[(1,)](
            a, b, c, d, e,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )