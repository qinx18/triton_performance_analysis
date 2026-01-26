import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, n, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask_i = i_idx < m
    
    # Load initial a values
    a_vals = tl.load(a_ptr + i_idx, mask=mask_i, other=0.0)
    
    # Sequential loop over j
    for j in range(m):
        # Calculate b index: i + m - j - 1
        b_idx = i_idx + m - j - 1
        mask_b = mask_i & (b_idx >= 0) & (b_idx < n)
        
        # Load values
        b_vals = tl.load(b_ptr + b_idx, mask=mask_b, other=0.0)
        c_val = tl.load(c_ptr + j)
        
        # Accumulate into a_vals
        a_vals = a_vals + b_vals * c_val
    
    # Store final result
    tl.store(a_ptr + i_idx, a_vals, mask=mask_i)

def s176_triton(a, b, c, m):
    n = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](
        a, b, c,
        n, m,
        BLOCK_SIZE=BLOCK_SIZE
    )