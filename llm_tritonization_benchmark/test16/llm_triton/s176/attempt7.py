import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < m
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Sequential loop over j
    for j in range(m):
        if j < m:
            # Load c[j] (scalar for all threads)
            c_j = tl.load(c_ptr + j)
            
            # Compute b indices: i+m-j-1
            b_indices = i_offsets + m - j - 1
            b_mask = mask & (b_indices >= 0) & (b_indices < (2 * m))
            
            # Load b[i+m-j-1]
            b_vals = tl.load(b_ptr + b_indices, mask=b_mask, other=0.0)
            
            # Accumulate: b[i+m-j-1] * c[j]
            acc += b_vals * c_j
    
    # Load current a values and add accumulator
    a_vals = tl.load(a_ptr + i_offsets, mask=mask, other=0.0)
    result = a_vals + acc
    
    # Store back to a
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s176_triton(a, b, c, m):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](
        a, b, c, m,
        BLOCK_SIZE=BLOCK_SIZE
    )