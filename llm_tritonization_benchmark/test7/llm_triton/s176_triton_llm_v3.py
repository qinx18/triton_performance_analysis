import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, LEN_1D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < m
    
    # Load initial a values
    a_vals = tl.load(a_ptr + i_offsets, mask=mask, other=0.0)
    
    # Loop over j
    for j in range(m):  # m = LEN_1D/2
        # Load c[j] (scalar)
        c_j = tl.load(c_ptr + j)
        
        # Load b values: b[i+m-j-1]
        b_indices = i_offsets + m - j - 1
        b_vals = tl.load(b_ptr + b_indices, mask=mask, other=0.0)
        
        # Update a[i] += b[i+m-j-1] * c[j]
        a_vals += b_vals * c_j
    
    # Store results
    tl.store(a_ptr + i_offsets, a_vals, mask=mask)

def s176_triton(a, b, c, m):
    LEN_1D = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](
        a, b, c,
        m, LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )