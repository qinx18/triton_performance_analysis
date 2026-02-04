import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(
    a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load values for first statement: a[i] = b[i] * c[i] * d[i]
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    
    # Compute and store a[i] = b[i] * c[i] * d[i]
    a_vals = b_vals * c_vals * d_vals
    tl.store(a_ptr + indices, a_vals, mask=mask)
    
    # Load a[i] from the original copy for second statement
    a_i_vals = tl.load(a_copy_ptr + indices, mask=mask)
    
    # Load a[i+1] from the original copy
    next_indices = indices + 1
    next_mask = next_indices < (n_elements + 1)
    a_next_vals = tl.load(a_copy_ptr + next_indices, mask=next_mask)
    
    # Compute b[i] = a[i] * a[i+1] * d[i]
    # Use the newly computed a[i] values, not from copy
    b_new_vals = a_vals * a_next_vals * d_vals
    
    # Store b[i] only where we have valid a[i+1]
    valid_mask = mask & next_mask
    tl.store(b_ptr + indices, b_new_vals, mask=valid_mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    
    # Create read-only copy of array a to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )