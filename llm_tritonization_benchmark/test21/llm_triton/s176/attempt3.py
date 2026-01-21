import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask_i = i_idx < m
    
    # Initialize accumulator with current a values
    a_vals = tl.load(a_ptr + i_idx, mask=mask_i, other=0.0)
    
    # Sequential loop over j (convolution pattern)
    for j in range(m):
        # Load c[j] (scalar broadcast)
        c_val = tl.load(c_ptr + j)
        
        # Calculate b indices: i+m-j-1
        b_idx = i_idx + m - j - 1
        mask_b = mask_i & (b_idx >= 0) & (b_idx < (m * 2))
        
        # Load b values
        b_vals = tl.load(b_ptr + b_idx, mask=mask_b, other=0.0)
        
        # Accumulate: a[i] += b[i+m-j-1] * c[j]
        a_vals = tl.where(mask_i, a_vals + b_vals * c_val, a_vals)
    
    # Store accumulated results
    tl.store(a_ptr + i_idx, a_vals, mask=mask_i)

def s176_triton(a, b, c):
    N = a.shape[0]
    m = N // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](a, b, c, m=m, BLOCK_SIZE=BLOCK_SIZE)