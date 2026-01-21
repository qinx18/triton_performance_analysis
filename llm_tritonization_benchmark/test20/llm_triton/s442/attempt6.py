import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load all arrays
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + idx, mask=mask, other=0.0)
    indx_vals = tl.load(indx_ptr + idx, mask=mask, other=0)
    
    # Compute updates based on indx values
    b_squared = b_vals * b_vals
    c_squared = c_vals * c_vals
    d_squared = d_vals * d_vals
    e_squared = e_vals * e_vals
    
    # Create masks for each case
    mask1 = (indx_vals == 1) & mask
    mask2 = (indx_vals == 2) & mask
    mask3 = (indx_vals == 3) & mask
    mask4 = (indx_vals == 4) & mask
    
    # Apply updates conditionally
    updates = tl.zeros_like(a_vals)
    updates = tl.where(mask1, b_squared, updates)
    updates = tl.where(mask2, c_squared, updates)
    updates = tl.where(mask3, d_squared, updates)
    updates = tl.where(mask4, e_squared, updates)
    
    # Update a values
    result = a_vals + updates
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )