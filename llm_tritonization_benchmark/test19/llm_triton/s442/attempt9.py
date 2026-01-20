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
    
    # Initialize result with original a values
    result = a_vals
    
    # Apply updates based on indx values - using exact switch logic
    # Case 1: a[i] += b[i] * b[i]
    mask_1 = (indx_vals == 1) & mask
    result = tl.where(mask_1, result + b_vals * b_vals, result)
    
    # Case 2: a[i] += c[i] * c[i]  
    mask_2 = (indx_vals == 2) & mask
    result = tl.where(mask_2, result + c_vals * c_vals, result)
    
    # Case 3: a[i] += d[i] * d[i]
    mask_3 = (indx_vals == 3) & mask
    result = tl.where(mask_3, result + d_vals * d_vals, result)
    
    # Case 4: a[i] += e[i] * e[i]
    mask_4 = (indx_vals == 4) & mask
    result = tl.where(mask_4, result + e_vals * e_vals, result)
    
    # Store result back to a
    tl.store(a_ptr + indices, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements, BLOCK_SIZE
    )