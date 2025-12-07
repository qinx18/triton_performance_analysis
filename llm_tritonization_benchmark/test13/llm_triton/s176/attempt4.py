import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Compute i indices for this block
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < m
    
    # Load initial a values
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
    
    # Sequential loop over j dimension
    for j in range(m):
        # Load c[j] (scalar broadcast)
        c_val = tl.load(c_ptr + j)
        
        # Compute b indices: i + m - j - 1
        b_idx = i_idx + m - j - 1
        
        # Load b values (no bounds check needed since b_idx will be in valid range)
        b_vals = tl.load(b_ptr + b_idx, mask=i_mask, other=0.0)
        
        # Compute: a[i] += b[i+m-j-1] * c[j]
        update = b_vals * c_val
        a_vals = a_vals + update
    
    # Store final a values
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)

def s176_triton(a, b, c, m):
    BLOCK_SIZE = 64
    
    # Launch single kernel with i-dimension parallelization
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](
        a, b, c, m,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a