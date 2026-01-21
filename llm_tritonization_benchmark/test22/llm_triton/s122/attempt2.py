import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the original loop
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate block range
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector for vectorized operations
    offsets = tl.arange(0, BLOCK_SIZE)
    iteration_indices = block_start + offsets
    
    # Mask for valid iterations
    mask = iteration_indices < num_iterations
    
    # Calculate i, j, k for valid iterations
    i = (n1 - 1) + iteration_indices * n3
    j = 1
    k = j * (iteration_indices + 1)  # k accumulates j for each iteration
    
    # Check bounds for array access
    valid_i = (i < N) & mask
    b_idx = N - k
    valid_b = (b_idx >= 0) & (b_idx < N)
    
    # Combined mask for safe memory access
    final_mask = valid_i & valid_b
    
    # Load values with masking
    a_vals = tl.load(a_ptr + i, mask=final_mask, other=0.0)
    b_vals = tl.load(b_ptr + b_idx, mask=final_mask, other=0.0)
    
    # Update a[i] with masking
    result = a_vals + b_vals
    tl.store(a_ptr + i, result, mask=final_mask)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the number of loop iterations
    if n1 - 1 >= N:
        return  # No iterations
    
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    if num_iterations <= 0:
        return
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    # Launch kernel
    s122_kernel[grid](
        a, b, n1, n3, N,
        BLOCK_SIZE=BLOCK_SIZE
    )