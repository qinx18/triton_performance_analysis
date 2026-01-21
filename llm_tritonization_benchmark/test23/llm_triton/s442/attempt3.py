import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load indices - ensure they are valid
    indices = tl.load(indx_ptr + offsets, mask=mask, other=0)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load all arrays
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared values
    b_squared = b_vals * b_vals
    c_squared = c_vals * c_vals
    d_squared = d_vals * d_vals
    e_squared = e_vals * e_vals
    
    # Create masks for each case
    case1_mask = (indices == 1) & mask
    case2_mask = (indices == 2) & mask
    case3_mask = (indices == 3) & mask
    case4_mask = (indices == 4) & mask
    
    # Initialize result with original a values
    result = a_vals
    
    # Apply updates based on switch cases - use tl.where to conditionally add
    result = tl.where(case1_mask, result + b_squared, result)
    result = tl.where(case2_mask, result + c_squared, result)
    result = tl.where(case3_mask, result + d_squared, result)
    result = tl.where(case4_mask, result + e_squared, result)
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )