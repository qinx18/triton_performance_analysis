import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Mask for valid elements (excluding last element since loop is i < n-1)
    mask = current_offsets < (n_elements - 1)
    
    # Load inputs
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask)
    
    # Compute values
    val_for_next = b_vals + e_vals  # a[i+1] = b[i] + e[i]
    val_for_curr = b_vals + c_vals  # a[i] = b[i] + c[i]
    
    # Store a[i] = b[i] + c[i] for all valid positions
    tl.store(a_ptr + current_offsets, val_for_curr, mask=mask)
    
    # Store a[i+1] = b[i] + e[i] for positions where i+1 is valid
    next_offsets = current_offsets + 1
    next_mask = mask & (next_offsets < n_elements)
    tl.store(a_ptr + next_offsets, val_for_next, mask=next_mask)

def s2244_triton(a, b, c, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )