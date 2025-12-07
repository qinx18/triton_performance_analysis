import torch
import triton
import triton.language as tl

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load index values
    indices = tl.load(indx_ptr + idx, mask=mask, other=0)
    
    # Load current a values
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    
    # Load array values
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + idx, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + idx, mask=mask, other=0.0)
    
    # Compute squared values
    b_sq = b_vals * b_vals
    c_sq = c_vals * c_vals
    d_sq = d_vals * d_vals
    e_sq = e_vals * e_vals
    
    # Apply conditional logic based on index values
    case1_mask = (indices == 1) & mask
    case2_mask = (indices == 2) & mask
    case3_mask = (indices == 3) & mask
    case4_mask = (indices == 4) & mask
    
    # Update a values based on cases
    a_vals = tl.where(case1_mask, a_vals + b_sq, a_vals)
    a_vals = tl.where(case2_mask, a_vals + c_sq, a_vals)
    a_vals = tl.where(case3_mask, a_vals + d_sq, a_vals)
    a_vals = tl.where(case4_mask, a_vals + e_sq, a_vals)
    
    # Store results
    tl.store(a_ptr + idx, a_vals, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )