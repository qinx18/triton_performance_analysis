import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask_i = i_idx < m
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load current a values
    a_vals = tl.load(a_ptr + i_idx, mask=mask_i, other=0.0)
    acc = a_vals
    
    # Loop over j sequentially
    for j in range(m):
        # Load c[j] (scalar broadcast)
        c_val = tl.load(c_ptr + j)
        
        # Calculate b indices: i + m - j - 1
        b_indices = i_idx + m - j - 1
        
        # Load b values
        b_vals = tl.load(b_ptr + b_indices, mask=mask_i, other=0.0)
        
        # Accumulate: a[i] += b[i+m-j-1] * c[j]
        acc += b_vals * c_val
    
    # Store final result
    tl.store(a_ptr + i_idx, acc, mask=mask_i)

def s176_triton(a, b, c, m):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](
        a, b, c, m,
        BLOCK_SIZE=BLOCK_SIZE
    )