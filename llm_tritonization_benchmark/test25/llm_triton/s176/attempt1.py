import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Calculate indices for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Mask for valid indices
    mask = i_indices < (n // 2)
    
    # Load current values of a[i]
    a_vals = tl.load(a_ptr + i_indices, mask=mask, other=0.0)
    
    m = n // 2
    
    # Sequential loop over j dimension (convolution pattern)
    for j in range(m):
        # Load c[j] (scalar broadcast)
        c_val = tl.load(c_ptr + j)
        
        # Calculate b indices: b[i+m-j-1]
        b_indices = i_indices + m - j - 1
        b_mask = mask & (b_indices >= 0) & (b_indices < n)
        
        # Load b values
        b_vals = tl.load(b_ptr + b_indices, mask=b_mask, other=0.0)
        
        # Accumulate: a[i] += b[i+m-j-1] * c[j]
        a_vals = a_vals + b_vals * c_val
    
    # Store results back to a
    tl.store(a_ptr + i_indices, a_vals, mask=mask)

def s176_triton(a, b, c):
    n = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n // 2, BLOCK_SIZE),)
    
    s176_kernel[grid](
        a, b, c, n,
        BLOCK_SIZE=BLOCK_SIZE
    )