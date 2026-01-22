import triton
import triton.language as tl
import torch

@triton.jit
def s131_kernel(a_ptr, a_copy_ptr, b_ptr, m, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n
    
    # Load from read-only copy for a[i + m]
    read_indices = indices + m
    read_mask = mask & (read_indices < (n + 1))
    a_read = tl.load(a_copy_ptr + read_indices, mask=read_mask, other=0.0)
    
    # Load b[i]
    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)
    
    # Compute result
    result = a_read + b_vals
    
    # Store to original array a[i]
    tl.store(a_ptr + indices, result, mask=mask)

def s131_triton(a, b, m):
    n = a.shape[0] - 1  # Loop runs from 0 to LEN_1D - 1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a, a_copy, b, m, n, BLOCK_SIZE=BLOCK_SIZE
    )