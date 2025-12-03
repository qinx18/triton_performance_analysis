import torch
import triton
import triton.language as tl

@triton.jit
def s277_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for start_idx in range(0, n_elements - 1, BLOCK_SIZE):
        current_offsets = start_idx + offsets
        mask = current_offsets < (n_elements - 1)
        
        # Load current values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Check conditions
        a_positive = a_vals >= 0.0
        b_positive = b_vals >= 0.0
        
        # Compute a[i] += c[i] * d[i] only when both conditions are false
        update_a_mask = mask & (~a_positive) & (~b_positive)
        new_a_vals = tl.where(update_a_mask, a_vals + c_vals * d_vals, a_vals)
        
        # Compute b[i+1] = c[i] + d[i] * e[i] when a[i] < 0
        compute_b_mask = mask & (~a_positive)
        b_result = c_vals + d_vals * e_vals
        
        # Store updated a values
        tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
        
        # Store b[i+1] values (need to handle i+1 indexing)
        b_next_offsets = current_offsets + 1
        b_next_mask = compute_b_mask & (b_next_offsets < n_elements)
        tl.store(b_ptr + b_next_offsets, b_result, mask=b_next_mask)

def s277_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create a copy of input arrays to avoid modifying originals
    a_work = a.clone()
    b_work = b.clone()
    
    grid = (1,)
    
    s277_kernel[grid](
        a_work, b_work, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a_work, b_work