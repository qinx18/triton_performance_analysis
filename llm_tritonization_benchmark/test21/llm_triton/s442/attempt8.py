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
    
    # Load arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    indx_vals = tl.load(indx_ptr + indices, mask=mask, other=0)
    
    # Initialize result with original a values
    result = a_vals
    
    # Apply updates based on indx values (only for valid indices)
    # Case 1: a[i] += b[i] * b[i]
    case1_mask = (indx_vals == 1) & mask
    result = result + tl.where(case1_mask, b_vals * b_vals, 0.0)
    
    # Case 2: a[i] += c[i] * c[i]
    case2_mask = (indx_vals == 2) & mask
    result = result + tl.where(case2_mask, c_vals * c_vals, 0.0)
    
    # Case 3: a[i] += d[i] * d[i]
    case3_mask = (indx_vals == 3) & mask
    result = result + tl.where(case3_mask, d_vals * d_vals, 0.0)
    
    # Case 4: a[i] += e[i] * e[i]
    case4_mask = (indx_vals == 4) & mask
    result = result + tl.where(case4_mask, e_vals * e_vals, 0.0)
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )