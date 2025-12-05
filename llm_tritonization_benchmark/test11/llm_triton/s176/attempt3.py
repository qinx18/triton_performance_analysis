import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + offsets
    i_mask = i_idx < m
    
    # Load initial values of a[i]
    a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
    
    # Sequential loop over j inside kernel
    for j in range(m):
        # Load c[j] (scalar)
        c_j = tl.load(c_ptr + j)
        
        # Compute b indices: i+m-j-1
        b_indices = i_idx + m - j - 1
        b_mask = i_mask & (b_indices >= 0) & (b_indices < 2*m)
        
        # Load b[i+m-j-1]
        b_vals = tl.load(b_ptr + b_indices, mask=b_mask, other=0.0)
        
        # Accumulate: a[i] += b[i+m-j-1] * c[j]
        increment = tl.where(b_mask, b_vals * c_j, 0.0)
        a_vals = a_vals + increment
    
    # Store final results
    tl.store(a_ptr + i_idx, a_vals, mask=i_mask)

def s176_triton(a, b, c, m):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](
        a, b, c, m,
        BLOCK_SIZE=BLOCK_SIZE
    )