import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the total number of valid indices
    num_indices = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get block start position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = block_start + offsets
    
    # Mask for valid indices within this block
    mask = block_offsets < num_indices
    
    # Convert block indices to actual array indices
    actual_indices = (n1 - 1) + block_offsets * n3
    
    # Additional mask to ensure we don't go beyond array bounds
    valid_mask = mask & (actual_indices < N)
    
    # Load values from arrays
    a_vals = tl.load(a_ptr + actual_indices, mask=valid_mask, other=0.0)
    b_vals = tl.load(b_ptr + actual_indices, mask=valid_mask, other=0.0)
    
    # Perform computation
    result = a_vals + b_vals
    
    # Store result back to array a
    tl.store(a_ptr + actual_indices, result, mask=valid_mask)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the total number of valid indices
    num_indices = (N - (n1 - 1) + n3 - 1) // n3
    
    if num_indices <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_indices, BLOCK_SIZE),)
    
    s172_kernel[grid](
        a, b, n1, n3, N,
        BLOCK_SIZE=BLOCK_SIZE
    )