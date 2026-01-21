import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load indices - ensure they are valid indices
    indices = tl.load(indx_ptr + offsets, mask=mask, other=0)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load arrays b, c, d, e
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared values
    b_squared = b_vals * b_vals
    c_squared = c_vals * c_vals
    d_squared = d_vals * d_vals
    e_squared = e_vals * e_vals
    
    # Apply conditional updates - only update where mask is True
    # Case 1: indx[i] == 1 -> a[i] += b[i] * b[i]
    case1_mask = (indices == 1) & mask
    a_vals = tl.where(case1_mask, a_vals + b_squared, a_vals)
    
    # Case 2: indx[i] == 2 -> a[i] += c[i] * c[i]
    case2_mask = (indices == 2) & mask
    a_vals = tl.where(case2_mask, a_vals + c_squared, a_vals)
    
    # Case 3: indx[i] == 3 -> a[i] += d[i] * d[i]
    case3_mask = (indices == 3) & mask
    a_vals = tl.where(case3_mask, a_vals + d_squared, a_vals)
    
    # Case 4: indx[i] == 4 -> a[i] += e[i] * e[i]
    case4_mask = (indices == 4) & mask
    a_vals = tl.where(case4_mask, a_vals + e_squared, a_vals)
    
    # Store results back
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s442_kernel[grid](a, b, c, d, e, indx, N, BLOCK_SIZE)