import torch
import triton
import triton.language as tl

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    indx_vals = tl.load(indx_ptr + idx, mask=mask)
    
    # Compute squared values
    b_squared = b_vals * b_vals
    c_squared = c_vals * c_vals
    d_squared = d_vals * d_vals
    e_squared = e_vals * e_vals
    
    # Apply switch logic using where operations
    result = a_vals
    result = tl.where(indx_vals == 1, a_vals + b_squared, result)
    result = tl.where(indx_vals == 2, a_vals + c_squared, result)
    result = tl.where(indx_vals == 3, a_vals + d_squared, result)
    result = tl.where(indx_vals == 4, a_vals + e_squared, result)
    
    # Store result
    tl.store(a_ptr + idx, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx, n_elements, BLOCK_SIZE
    )
    
    return a