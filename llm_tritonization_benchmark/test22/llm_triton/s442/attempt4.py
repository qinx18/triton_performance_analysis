import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load indices
    indices = tl.load(indx_ptr + offsets, mask=mask, other=0)
    
    # Load current values of a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Load arrays b, c, d, e
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Compute updates for each case
    update_val = tl.where(indices == 1, b_vals * b_vals,
                 tl.where(indices == 2, c_vals * c_vals,
                 tl.where(indices == 3, d_vals * d_vals,
                 tl.where(indices == 4, e_vals * e_vals, 0.0))))
    
    # Apply update only for valid indices (1-4)
    valid_idx = (indices >= 1) & (indices <= 4) & mask
    a_vals = tl.where(valid_idx, a_vals + update_val, a_vals)
    
    # Store results back
    tl.store(a_ptr + offsets, a_vals, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s442_kernel[grid](a, b, c, d, e, indx, N, BLOCK_SIZE)