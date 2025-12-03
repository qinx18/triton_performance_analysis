import torch
import triton
import triton.language as tl

@triton.jit
def s1244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Current indices
    i_offsets = block_start + offsets
    i_mask = i_offsets < n
    
    # Next indices for a[i+1]
    i_plus_1_offsets = i_offsets + 1
    i_plus_1_mask = i_plus_1_offsets < (n + 1)  # a has n+1 elements total
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + i_offsets, mask=i_mask)
    c_vals = tl.load(c_ptr + i_offsets, mask=i_mask)
    
    # Compute a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i]
    a_vals = b_vals + c_vals * c_vals + b_vals * b_vals + c_vals
    
    # Store a[i]
    tl.store(a_ptr + i_offsets, a_vals, mask=i_mask)
    
    # Load a[i+1] for d[i] computation
    a_plus_1_vals = tl.load(a_ptr + i_plus_1_offsets, mask=i_plus_1_mask)
    
    # Compute d[i] = a[i] + a[i+1]
    d_vals = a_vals + a_plus_1_vals
    
    # Store d[i]
    tl.store(d_ptr + i_offsets, d_vals, mask=i_mask)

def s1244_triton(a, b, c, d):
    n = a.shape[0] - 1  # Process n-1 elements (0 to n-2)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s1244_kernel[grid](
        a, b, c, d, n, BLOCK_SIZE=BLOCK_SIZE
    )