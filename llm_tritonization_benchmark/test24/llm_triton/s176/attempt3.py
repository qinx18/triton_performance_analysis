import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < m
    
    # Load initial a values
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
    
    # Sequential loop over j
    for j in range(m):
        # Load c[j] (scalar broadcast)
        c_val = tl.load(c_ptr + j)
        
        # Calculate b indices: i + m - j - 1
        b_idx = i_idx + m - j - 1
        
        # Load b values with proper masking
        b_mask = i_mask & (b_idx >= 0)
        b_vals = tl.load(b_ptr + b_idx, mask=b_mask, other=0.0)
        
        # Accumulate: a[i] += b[i+m-j-1] * c[j]
        a_vals += b_vals * c_val
    
    # Store final results
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](a, b, c, m, BLOCK_SIZE=BLOCK_SIZE)