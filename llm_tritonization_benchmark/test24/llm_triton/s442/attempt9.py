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
    
    # Load indices and arrays
    indx_vals = tl.load(indx_ptr + indices, mask=mask, other=0)
    a_vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + indices, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + indices, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + indices, mask=mask, other=0.0)
    
    # Create masks for each case
    mask_1 = (indx_vals == 1) & mask
    mask_2 = (indx_vals == 2) & mask
    mask_3 = (indx_vals == 3) & mask
    mask_4 = (indx_vals == 4) & mask
    
    # Compute updates for each case
    update_1 = a_vals + b_vals * b_vals
    update_2 = a_vals + c_vals * c_vals
    update_3 = a_vals + d_vals * d_vals
    update_4 = a_vals + e_vals * e_vals
    
    # Apply the correct update based on indx value
    result = a_vals
    result = tl.where(mask_1, update_1, result)
    result = tl.where(mask_2, update_2, result)
    result = tl.where(mask_3, update_3, result)
    result = tl.where(mask_4, update_4, result)
    
    # Store result back
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