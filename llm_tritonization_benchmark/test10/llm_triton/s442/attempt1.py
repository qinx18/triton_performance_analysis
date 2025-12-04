import torch
import triton
import triton.language as tl

@triton.jit
def s442_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load index values
    indx_vals = tl.load(indx_ptr + idx, mask=mask, other=0)
    
    # Load array values
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + idx, mask=mask, other=0.0)
    
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
    
    # Apply updates based on index values
    update_vals = tl.where(case1_mask, b_sq,
                  tl.where(case2_mask, c_sq,
                  tl.where(case3_mask, d_sq,
                  tl.where(case4_mask, e_sq, 0.0))))
    
    result = a_vals + update_vals
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a