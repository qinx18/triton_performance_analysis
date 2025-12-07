import torch
import triton
import triton.language as tl

@triton.jit
def s1244_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    
    # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
    
    # Store a[i]
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    
    # For d[i] = a[i] + a[i+1], we need a[i] from copy and a[i+1] from copy
    a_i_vals = tl.load(a_copy_ptr + current_offsets, mask=mask)
    next_offsets = current_offsets + 1
    next_mask = next_offsets < n_elements
    a_i_plus_1_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
    
    # Compute d[i] = a[i] + a[i+1]
    d_vals = a_i_vals + a_i_plus_1_vals
    
    # Store d[i] (only where both current and next are valid)
    valid_mask = mask & next_mask
    tl.store(d_ptr + current_offsets, d_vals, mask=valid_mask)

def s1244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Process LEN_1D-1 elements
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1244_kernel[grid](
        a, a_copy, b, c, d, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )