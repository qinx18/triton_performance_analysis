import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load indices
    indx_vals = tl.load(indx_ptr + indices, mask=mask, other=0)
    
    # Load array values
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + indices, mask=mask, other=0.0)
    
    # Compute squares
    b_squared = b_vals * b_vals
    c_squared = c_vals * c_vals
    d_squared = d_vals * d_vals
    e_squared = e_vals * e_vals
    
    # Create masks for each case
    mask_case1 = (indx_vals == 1) & mask
    mask_case2 = (indx_vals == 2) & mask
    mask_case3 = (indx_vals == 3) & mask
    mask_case4 = (indx_vals == 4) & mask
    
    # Apply updates based on switch cases
    result = a_vals
    result = tl.where(mask_case1, result + b_squared, result)
    result = tl.where(mask_case2, result + c_squared, result)
    result = tl.where(mask_case3, result + d_squared, result)
    result = tl.where(mask_case4, result + e_squared, result)
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )