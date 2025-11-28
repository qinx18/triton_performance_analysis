import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(
    a_ptr,
    b_ptr,
    LEN_1D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate the range of indices this block will handle
    block_start = pid * BLOCK_SIZE + 1  # Start from 1 (not 0)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to ensure we don't go beyond LEN_1D and start from index 1
    mask = offsets < LEN_1D
    
    # Load a[0] (scalar broadcast)
    a0 = tl.load(a_ptr)
    
    # Load b[i] for the current block
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    LEN_1D = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size (excluding index 0 since we start from index 1)
    grid_size = triton.cdiv(LEN_1D - 1, BLOCK_SIZE)
    
    # Launch kernel
    s113_kernel[(grid_size,)](
        a, b,
        LEN_1D=LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a