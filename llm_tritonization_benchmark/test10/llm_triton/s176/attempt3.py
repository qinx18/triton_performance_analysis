import triton
import triton.language as tl
import torch

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate i indices for this block
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < m
    
    # Sequential loop over j (in-kernel)
    for j in range(m):
        # Load c[j] (scalar broadcast)
        c_j = tl.load(c_ptr + j)
        
        # Calculate b indices: b[i+m-j-1]
        b_indices = i_idx + m - j - 1
        b_mask = i_mask & (b_indices >= 0)
        
        # Load current a[i] values
        a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
        
        # Load b[i+m-j-1] values
        b_vals = tl.load(b_ptr + b_indices, mask=b_mask, other=0.0)
        
        # Compute: a[i] += b[i+m-j-1] * c[j]
        result = a_vals + b_vals * c_j
        
        # Store back to a[i]
        tl.store(a_ptr + i_idx, result, mask=i_mask)

def s176_triton(a, b, c, m):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](
        a, b, c, m,
        BLOCK_SIZE=BLOCK_SIZE
    )