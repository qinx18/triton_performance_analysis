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
    mask = (block_start + offsets) < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask)
    indx_vals = tl.load(indx_ptr + block_start + offsets, mask=mask)
    
    # Compute updates based on switch logic
    case1_mask = (indx_vals == 1) & mask
    case2_mask = (indx_vals == 2) & mask
    case3_mask = (indx_vals == 3) & mask
    case4_mask = (indx_vals == 4) & mask
    
    update_vals = tl.zeros_like(a_vals)
    update_vals = tl.where(case1_mask, b_vals * b_vals, update_vals)
    update_vals = tl.where(case2_mask, c_vals * c_vals, update_vals)
    update_vals = tl.where(case3_mask, d_vals * d_vals, update_vals)
    update_vals = tl.where(case4_mask, e_vals * e_vals, update_vals)
    
    # Apply updates
    result = a_vals + update_vals
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )