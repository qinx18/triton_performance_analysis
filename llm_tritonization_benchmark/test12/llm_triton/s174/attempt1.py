import torch
import triton
import triton.language as tl

@triton.jit
def s174_kernel(a_ptr, b_ptr, M, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices
    mask = indices < M
    
    # Load data from arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Perform computation: a[i+M] = a[i] + b[i]
    result = a_vals + b_vals
    
    # Store result at offset indices (i+M)
    output_indices = indices + M
    tl.store(a_ptr + output_indices, result, mask=mask)

def s174_triton(a, b, M):
    # Define block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(M, BLOCK_SIZE)
    
    # Launch kernel
    s174_kernel[(grid_size,)](
        a, b, M,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a