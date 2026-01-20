import triton
import triton.language as tl
import torch

@triton.jit
def s131_kernel(
    a_ptr, a_copy_ptr, b_ptr,
    n, m,
    BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load from read-only copy for a[i + m]
        read_offsets = current_offsets + m
        read_mask = mask & (read_offsets < (n + m))
        a_vals = tl.load(a_copy_ptr + read_offsets, mask=read_mask, other=0.0)
        
        # Load b[i]
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute result
        result = a_vals + b_vals
        
        # Store to original array
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s131_triton(a, b, m):
    n = a.shape[0] - 1  # LEN_1D - 1
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a, a_copy, b,
        n, m,
        BLOCK_SIZE=BLOCK_SIZE
    )