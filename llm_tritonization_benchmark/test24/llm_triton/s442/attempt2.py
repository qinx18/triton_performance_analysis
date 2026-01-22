import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load indices - ensure they are in valid range
    indices = tl.load(indx_ptr + idx, mask=mask, other=0)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Load values from other arrays
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + idx, mask=mask, other=0.0)
    
    # Compute updates based on switch/goto logic
    # Only update if mask is true and index is valid (1-4)
    case1_mask = mask & (indices == 1)
    case2_mask = mask & (indices == 2)
    case3_mask = mask & (indices == 3)
    case4_mask = mask & (indices == 4)
    
    # Apply updates conditionally
    update = tl.where(case1_mask, b_vals * b_vals,
             tl.where(case2_mask, c_vals * c_vals,
             tl.where(case3_mask, d_vals * d_vals,
             tl.where(case4_mask, e_vals * e_vals, 0.0))))
    
    # Update a values
    new_a_vals = a_vals + update
    
    # Store results only where mask is true
    tl.store(a_ptr + idx, new_a_vals, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )