import torch
import triton
import triton.language as tl

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load indices
    indices = tl.load(indx_ptr + block_start + offsets, mask=mask, other=0)
    
    # Load all arrays
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Switch-case logic using conditional masks
    case1_mask = (indices == 1)
    case2_mask = (indices == 2)
    case3_mask = (indices == 3)
    case4_mask = (indices == 4)
    
    # Compute updates based on cases
    update1 = b_vals * b_vals
    update2 = c_vals * c_vals
    update3 = d_vals * d_vals
    update4 = e_vals * e_vals
    
    # Apply updates conditionally
    result = a_vals
    result = tl.where(case1_mask, result + update1, result)
    result = tl.where(case2_mask, result + update2, result)
    result = tl.where(case3_mask, result + update3, result)
    result = tl.where(case4_mask, result + update4, result)
    
    # Store result
    tl.store(a_ptr + block_start + offsets, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a