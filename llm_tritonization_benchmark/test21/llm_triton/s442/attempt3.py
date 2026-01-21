import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load index values
    indx_vals = tl.load(indx_ptr + offsets, mask=mask, other=0)
    
    # Load all input arrays
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Compute squared values
    b_sq = b_vals * b_vals
    c_sq = c_vals * c_vals  
    d_sq = d_vals * d_vals
    e_sq = e_vals * e_vals
    
    # Add the appropriate squared value based on index using separate where conditions
    result = a_vals + tl.where(indx_vals == 1, b_sq, 0.0)
    result = result + tl.where(indx_vals == 2, c_sq, 0.0)
    result = result + tl.where(indx_vals == 3, d_sq, 0.0)
    result = result + tl.where(indx_vals == 4, e_sq, 0.0)
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s442_kernel[grid](a, b, c, d, e, indx, N, BLOCK_SIZE=BLOCK_SIZE)