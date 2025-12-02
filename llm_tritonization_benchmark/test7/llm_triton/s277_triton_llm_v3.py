import torch
import triton
import triton.language as tl

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask)
    
    # Load b[i+1] for the b_vals update (need to handle boundary)
    b_next_mask = (block_start + offsets + 1) < (n_elements + 1)
    
    # Control flow logic
    a_positive = a_vals >= 0.0
    b_positive = b_vals >= 0.0
    
    # If a[i] >= 0, skip everything (goto L20)
    # If b[i] >= 0, skip a[i] update but do b[i+1] update (goto L30)
    # Otherwise, do a[i] update and b[i+1] update
    
    # Update a[i] only if both a[i] < 0 and b[i] < 0
    update_a_mask = (~a_positive) & (~b_positive)
    new_a = tl.where(update_a_mask, a_vals + c_vals * d_vals, a_vals)
    
    # Update b[i+1] if a[i] < 0 (regardless of b[i])
    update_b_next = ~a_positive
    b_next_val = c_vals + d_vals * e_vals
    
    # Store updated a values
    tl.store(a_ptr + block_start + offsets, new_a, mask=mask)
    
    # Store updated b[i+1] values
    next_offsets = block_start + offsets + 1
    next_mask = (next_offsets < (n_elements + 1)) & update_b_next & mask
    tl.store(b_ptr + next_offsets, b_next_val, mask=next_mask)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s277_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )