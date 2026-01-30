import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, n, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    mask = i_indices < m
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load initial a values
    a_vals = tl.load(a_ptr + i_indices, mask=mask, other=0.0)
    acc = a_vals
    
    # Sequential loop over j
    for j in range(m):
        # Load c[j] (scalar for all threads)
        c_j = tl.load(c_ptr + j)
        
        # Compute b indices: i + m - j - 1
        b_indices = i_indices + m - j - 1
        b_mask = mask & (b_indices < n) & (b_indices >= 0)
        
        # Load b values
        b_vals = tl.load(b_ptr + b_indices, mask=b_mask, other=0.0)
        
        # Accumulate: a[i] += b[i+m-j-1] * c[j]
        acc += b_vals * c_j
    
    # Store results
    tl.store(a_ptr + i_indices, acc, mask=mask)

def s176_triton(a, b, c):
    n = a.shape[0]
    m = n // 2
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    s176_kernel[grid](
        a, b, c,
        n, m,
        BLOCK_SIZE=BLOCK_SIZE
    )