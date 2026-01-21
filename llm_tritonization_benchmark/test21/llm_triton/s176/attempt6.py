import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask = i_idx < m
    
    # Load initial values of a
    a_vals = tl.load(a_ptr + i_idx, mask=mask, other=0.0)
    
    # Sequential loop over j
    for j in range(m):
        # Load c[j] - broadcast to all threads
        c_val = tl.load(c_ptr + j)
        
        # Compute b indices: i+m-j-1
        b_idx = i_idx + m - j - 1
        b_mask = mask & (b_idx >= 0)
        
        # Load b values
        b_vals = tl.load(b_ptr + b_idx, mask=b_mask, other=0.0)
        
        # Accumulate: a[i] += b[i+m-j-1] * c[j]
        a_vals += b_vals * c_val
    
    # Store accumulated results
    tl.store(a_ptr + i_idx, a_vals, mask=mask)

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](a, b, c, m=m, BLOCK_SIZE=BLOCK_SIZE)