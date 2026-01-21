import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + block_start
    mask = offsets < N
    
    # Load indices - ensure they're within valid range for switch cases
    indices = tl.load(indx_ptr + offsets, mask=mask, other=0)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load input arrays only where mask is true
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squares for all arrays
    b_sq = b_vals * b_vals
    c_sq = c_vals * c_vals  
    d_sq = d_vals * d_vals
    e_sq = e_vals * e_vals
    
    # Create masks for each case - only process valid indices
    valid_mask = mask
    case1_mask = valid_mask & (indices == 1)
    case2_mask = valid_mask & (indices == 2)
    case3_mask = valid_mask & (indices == 3)
    case4_mask = valid_mask & (indices == 4)
    
    # Apply the switch logic: a[i] += corresponding_square
    result = a_vals
    result = tl.where(case1_mask, result + b_sq, result)
    result = tl.where(case2_mask, result + c_sq, result)
    result = tl.where(case3_mask, result + d_sq, result)
    result = tl.where(case4_mask, result + e_sq, result)
    
    # Store result only where mask is true
    tl.store(a_ptr + offsets, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s442_kernel[grid](a, b, c, d, e, indx, N, BLOCK_SIZE)