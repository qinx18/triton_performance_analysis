import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, n, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate i offsets for this block
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < m
    
    # Sequential loop over j dimension (required for accumulation)
    for j in range(m):
        if i_mask.any():
            # Calculate b indices: i+m-j-1
            b_indices = i_idx + m - j - 1
            b_mask = i_mask & (b_indices >= 0) & (b_indices < n)
            
            # Load current a values
            a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
            
            # Load b values with shifted indices
            b_vals = tl.load(b_ptr + b_indices, mask=b_mask, other=0.0)
            
            # Load scalar c[j] and broadcast
            c_val = tl.load(c_ptr + j)
            
            # Compute update: a[i] += b[i+m-j-1] * c[j]
            update = b_vals * c_val
            new_a_vals = a_vals + update
            
            # Store back to a
            tl.store(a_ptr + i_idx, new_a_vals, mask=i_mask)

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](
        a, b, c, n, m, BLOCK_SIZE
    )