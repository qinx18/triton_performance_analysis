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
    
    # Load data
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    indx_vals = tl.load(indx_ptr + indices, mask=mask)
    
    # Compute squared values
    b_squared = b_vals * b_vals
    c_squared = c_vals * c_vals
    d_squared = d_vals * d_vals
    e_squared = e_vals * e_vals
    
    # Create masks for each case
    mask_1 = (indx_vals == 1) & mask
    mask_2 = (indx_vals == 2) & mask
    mask_3 = (indx_vals == 3) & mask
    mask_4 = (indx_vals == 4) & mask
    
    # Compute updates for each case
    result = a_vals
    result = tl.where(mask_1, a_vals + b_squared, result)
    result = tl.where(mask_2, a_vals + c_squared, result)
    result = tl.where(mask_3, a_vals + d_squared, result)
    result = tl.where(mask_4, a_vals + e_squared, result)
    
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