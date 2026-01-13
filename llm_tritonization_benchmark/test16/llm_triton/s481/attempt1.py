import torch
import triton
import triton.language as tl

@triton.jit
def s481_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Check if any element in this block has d[i] < 0
    # If so, we need to handle partial processing
    has_negative = d_vals < 0.0
    
    # Find first negative index in this block
    negative_mask = has_negative & mask
    
    # Only process elements before any negative value
    # Use cumulative approach to mask out elements after first negative
    valid_mask = mask
    
    # For each position, check if there's a negative value at or before it
    for offset in range(BLOCK_SIZE):
        if offset > 0:
            # Check if previous position had negative value
            prev_negative = tl.where(offsets == (offset - 1), has_negative, False)
            if tl.sum(prev_negative.to(tl.int32)) > 0:
                # Mask out current and subsequent positions
                current_invalid = offsets >= offset
                valid_mask = valid_mask & (~current_invalid)
    
    # Also mask out the position with negative value itself
    valid_mask = valid_mask & (~has_negative)
    
    # Load and compute only for valid positions
    a_vals = tl.load(a_ptr + indices, mask=valid_mask)
    b_vals = tl.load(b_ptr + indices, mask=valid_mask)
    c_vals = tl.load(c_ptr + indices, mask=valid_mask)
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store result
    tl.store(a_ptr + indices, result, mask=valid_mask)

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Check for early exit condition sequentially first
    negative_mask = d < 0.0
    if torch.any(negative_mask):
        # Find first negative index
        first_negative_idx = torch.argmax(negative_mask.int()).item()
        # Only process elements before the first negative
        if first_negative_idx > 0:
            a[:first_negative_idx] += b[:first_negative_idx] * c[:first_negative_idx]
        return
    
    # No early exit needed, process all elements with Triton
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s481_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )