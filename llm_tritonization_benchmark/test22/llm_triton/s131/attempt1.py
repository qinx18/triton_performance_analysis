import triton
import triton.language as tl
import torch

@triton.jit
def s131_kernel(a_ptr, a_copy_ptr, b_ptr, n, m, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    i_offsets = block_start + offsets
    mask = i_offsets < n
    
    # Read from copy (a_copy[i + m]) and original b[i]
    a_read_offsets = i_offsets + m
    a_read_mask = a_read_offsets < (n + m)  # Since we read from i+m, need bounds check
    
    a_vals = tl.load(a_copy_ptr + a_read_offsets, mask=mask & a_read_mask, other=0.0)
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    
    # Compute a[i] = a[i + m] + b[i]
    result = a_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + i_offsets, result, mask=mask)

def s131_triton(a, b, m):
    n = a.shape[0] - 1  # LEN_1D - 1
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a, a_copy, b, n, m,
        BLOCK_SIZE=BLOCK_SIZE
    )