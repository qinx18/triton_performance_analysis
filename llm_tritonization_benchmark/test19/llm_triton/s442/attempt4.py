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
    
    # Create masks for each case
    case1_mask = (indx_vals == 1) & mask
    case2_mask = (indx_vals == 2) & mask
    case3_mask = (indx_vals == 3) & mask
    case4_mask = (indx_vals == 4) & mask
    
    # Initialize result with current a values
    result = a_vals
    
    # Apply updates - use where to conditionally add
    result = tl.where(case1_mask, result + b_vals * b_vals, result)
    result = tl.where(case2_mask, result + c_vals * c_vals, result)
    result = tl.where(case3_mask, result + d_vals * d_vals, result)
    result = tl.where(case4_mask, result + e_vals * e_vals, result)
    
    # Store result only where mask is valid
    tl.store(a_ptr + indices, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements, BLOCK_SIZE
    )