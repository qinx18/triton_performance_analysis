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
    
    # Load indices
    indices = tl.load(indx_ptr + idx, mask=mask, other=0)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Load values from other arrays
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + idx, mask=mask, other=0.0)
    
    # Compute updates based on switch logic
    case1_mask = (indices == 1) & mask
    case2_mask = (indices == 2) & mask
    case3_mask = (indices == 3) & mask 
    case4_mask = (indices == 4) & mask
    
    # Apply conditional updates
    new_a_vals = tl.where(case1_mask, a_vals + b_vals * b_vals, a_vals)
    new_a_vals = tl.where(case2_mask, new_a_vals + c_vals * c_vals, new_a_vals)
    new_a_vals = tl.where(case3_mask, new_a_vals + d_vals * d_vals, new_a_vals)
    new_a_vals = tl.where(case4_mask, new_a_vals + e_vals * e_vals, new_a_vals)
    
    # Store results
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