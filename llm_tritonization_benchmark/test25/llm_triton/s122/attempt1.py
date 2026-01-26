import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N):
    # Calculate the number of iterations in the original loop
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3  # ceiling division
    
    # Get block index
    block_idx = tl.program_id(0)
    
    # Check if this block is within valid range
    if block_idx >= num_iterations:
        return
    
    # Calculate the i value for this iteration
    i = (n1 - 1) + block_idx * n3
    
    # Calculate k value: k starts at 0, j starts at 1
    # After block_idx iterations: k = j * (block_idx + 1) where j = 1
    k = block_idx + 1
    
    # Calculate indices
    b_idx = N - k
    
    # Check bounds
    if i < N and b_idx >= 0:
        # Load values
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + b_idx)
        
        # Update a[i]
        result = a_val + b_val
        tl.store(a_ptr + i, result)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the number of iterations needed
    if n1 - 1 >= N:
        return  # No iterations needed
    
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3  # ceiling division
    
    if num_iterations <= 0:
        return
    
    # Launch kernel with one thread per iteration
    grid = (num_iterations,)
    s122_kernel[grid](a, b, n1, n3, N)