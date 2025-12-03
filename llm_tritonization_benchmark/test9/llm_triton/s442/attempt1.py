import torch
import triton
import triton.language as tl

@triton.jit
def s442_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load index values
    indx_vals = tl.load(indx_ptr + block_start + offsets, mask=mask, other=0)
    
    # Load current a values
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Load all input arrays
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Compute squared values
    b_sq = b_vals * b_vals
    c_sq = c_vals * c_vals
    d_sq = d_vals * d_vals
    e_sq = e_vals * e_vals
    
    # Create masks for each case
    case1_mask = indx_vals == 1
    case2_mask = indx_vals == 2
    case3_mask = indx_vals == 3
    case4_mask = indx_vals == 4
    
    # Compute updates for each case
    update1 = tl.where(case1_mask, b_sq, 0.0)
    update2 = tl.where(case2_mask, c_sq, 0.0)
    update3 = tl.where(case3_mask, d_sq, 0.0)
    update4 = tl.where(case4_mask, e_sq, 0.0)
    
    # Sum all updates
    total_update = update1 + update2 + update3 + update4
    
    # Update a values
    result = a_vals + total_update
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a