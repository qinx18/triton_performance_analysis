import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < m
    
    for j in range(m):
        # Load current a values
        a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
        
        # Load c[j] as scalar
        c_val = tl.load(c_ptr + j)
        
        # Calculate b indices: i+m-j-1
        b_idx = i_idx + m - j - 1
        b_mask = i_mask & (b_idx >= 0) & (b_idx < (2 * m))
        
        # Load b values with bounds check
        b_vals = tl.load(b_ptr + b_idx, mask=b_mask, other=0.0)
        
        # Compute update and store
        result = a_vals + b_vals * c_val
        tl.store(a_ptr + i_idx, result, mask=i_mask)

def s176_triton(a, b, c, m):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](
        a, b, c, m,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a