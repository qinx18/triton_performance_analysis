import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load index values
    indx_vals = tl.load(indx_ptr + indices, mask=mask, other=0)
    
    # Load current a values
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Load other array values
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + indices, mask=mask, other=0.0)
    
    # Create masks for each case - only update when indx value matches
    case1_mask = (indx_vals == 1) & mask
    case2_mask = (indx_vals == 2) & mask
    case3_mask = (indx_vals == 3) & mask
    case4_mask = (indx_vals == 4) & mask
    
    # Apply updates conditionally - each element gets exactly one update based on its indx value
    update_vals = tl.where(case1_mask, b_vals * b_vals,
                  tl.where(case2_mask, c_vals * c_vals,
                  tl.where(case3_mask, d_vals * d_vals,
                  tl.where(case4_mask, e_vals * e_vals, 0.0))))
    
    # Only add update if any case matched
    any_case_mask = case1_mask | case2_mask | case3_mask | case4_mask
    result = tl.where(any_case_mask, a_vals + update_vals, a_vals)
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements, BLOCK_SIZE
    )