import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offset = pid * BLOCK_SIZE
    i_offsets = i_offset + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < m
    
    # Load initial a values
    a_vals = tl.load(a_ptr + i_offsets, mask=i_mask, other=0.0)
    
    # Sequential loop over j
    for j in range(m):
        # Calculate b indices: i + m - j - 1
        b_indices = i_offsets + m - j - 1
        b_vals = tl.load(b_ptr + b_indices, mask=i_mask, other=0.0)
        
        # Load c[j] (scalar)
        c_val = tl.load(c_ptr + j)
        
        # Accumulate: a[i] += b[i+m-j-1] * c[j]
        a_vals += b_vals * c_val
    
    # Store final result
    tl.store(a_ptr + i_offsets, a_vals, mask=i_mask)

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](a, b, c, m, BLOCK_SIZE=BLOCK_SIZE)