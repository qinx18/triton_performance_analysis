import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting index for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Generate indices for this block
    block_indices = tl.arange(0, BLOCK_SIZE)
    
    # Calculate the actual indices following the pattern: i = n1-1; i < LEN_1D; i += n3
    start_idx = n1 - 1
    indices = start_idx + (block_start + block_indices) * n3
    
    # Create mask to ensure we don't go out of bounds
    mask = indices < LEN_1D
    
    # Load values from arrays a and b
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Perform the computation: a[i] += b[i]
    result = a_vals + b_vals
    
    # Store the result back to array a
    tl.store(a_ptr + indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate total number of iterations in the original loop
    if n3 <= 0:
        return  # No valid iterations
    
    start_idx = n1 - 1
    if start_idx >= LEN_1D:
        return  # No valid iterations
    
    total_iterations = (LEN_1D - start_idx + n3 - 1) // n3
    
    if total_iterations <= 0:
        return
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = (total_iterations + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    s172_kernel[(num_blocks,)](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )