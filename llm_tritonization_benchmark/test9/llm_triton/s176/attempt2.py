import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    j = tl.program_id(0)
    
    # Define i offsets once
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Loop over i dimension in blocks
    for i_block_start in range(0, m, BLOCK_SIZE):
        i_indices = i_block_start + i_offsets
        i_mask = i_indices < m
        
        # Load a[i]
        a_vals = tl.load(a_ptr + i_indices, mask=i_mask, other=0.0)
        
        # Calculate b index: i + m - j - 1
        b_indices = i_indices + m - j - 1
        
        # Load b[i + m - j - 1]
        b_vals = tl.load(b_ptr + b_indices, mask=i_mask, other=0.0)
        
        # Load c[j] (scalar)
        c_val = tl.load(c_ptr + j)
        
        # Compute a[i] += b[i + m - j - 1] * c[j]
        result = a_vals + b_vals * c_val
        
        # Store back to a[i]
        tl.store(a_ptr + i_indices, result, mask=i_mask)

def s176_triton(a, b, c, m):
    # Grid size: one block per j iteration
    grid = (m,)
    
    # Block size for i dimension
    BLOCK_SIZE = 128
    
    # Launch kernel
    s176_kernel[grid](
        a, b, c,
        m,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a