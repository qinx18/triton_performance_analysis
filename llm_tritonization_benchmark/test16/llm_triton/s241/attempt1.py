import torch
import triton
import triton.language as tl

@triton.jit
def s241_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    
    # First statement: a[i] = b[i] * c[i] * d[i]
    a_vals = b_vals * c_vals * d_vals
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    
    # Load a_copy[i+1] for second statement
    next_offsets = current_offsets + 1
    next_mask = next_offsets < (n_elements + 1)  # Allow loading up to original array size
    a_next_vals = tl.load(a_copy_ptr + next_offsets, mask=next_mask)
    
    # Second statement: b[i] = a[i] * a[i+1] * d[i]
    b_new_vals = a_vals * a_next_vals * d_vals
    tl.store(b_ptr + current_offsets, b_new_vals, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, a_copy, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )