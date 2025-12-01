import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which j iteration this program handles
    j = pid
    if j >= m:
        return
    
    # Pre-define offsets for vectorized operations
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process array a in blocks
    for block_start in range(0, m, BLOCK_SIZE):
        # Calculate current offsets for this block
        a_offsets = block_start + offsets
        a_mask = a_offsets < m
        
        # Load current values from a
        a_vals = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        
        # Calculate b indices: i + m - j - 1
        b_offsets = a_offsets + m - j - 1
        b_mask = (a_offsets < m) & (b_offsets >= 0) & (b_offsets < LEN_1D)
        
        # Load values from b
        b_vals = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)
        
        # Load c[j] (scalar broadcast)
        c_val = tl.load(c_ptr + j)
        
        # Compute update: b[i+m-j-1] * c[j]
        update = b_vals * c_val
        
        # Add to a values
        new_a_vals = a_vals + update
        
        # Store back to a
        tl.store(a_ptr + a_offsets, new_a_vals, mask=a_mask)

def s176_triton(a, b, c):
    LEN_1D = a.shape[0]
    m = LEN_1D // 2
    
    # Launch kernel with one thread per j iteration
    BLOCK_SIZE = 256
    grid = (m,)
    
    s176_kernel[grid](a, b, c, m, LEN_1D, BLOCK_SIZE)