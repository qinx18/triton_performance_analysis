import torch
import triton
import triton.language as tl

@triton.jit
def s176_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = block_start + offsets
    mask = i_idx < N // 2
    
    m = N // 2
    
    # Load initial values of a
    a_vals = tl.load(a_ptr + i_idx, mask=mask, other=0.0)
    
    # Sequential loop over j
    for j in range(N // 2):
        # Load c[j] - broadcast to all elements
        c_val = tl.load(c_ptr + j)
        
        # Compute b indices: i + m - j - 1
        b_indices = i_idx + m - j - 1
        b_mask = mask & (b_indices >= 0) & (b_indices < N)
        
        # Load b values
        b_vals = tl.load(b_ptr + b_indices, mask=b_mask, other=0.0)
        
        # Update accumulator
        a_vals += b_vals * c_val
    
    # Store final results
    tl.store(a_ptr + i_idx, a_vals, mask=mask)

def s176_triton(a, b, c):
    N = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    
    s176_kernel[grid](a, b, c, N, BLOCK_SIZE)