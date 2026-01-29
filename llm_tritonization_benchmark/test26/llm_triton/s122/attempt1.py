import torch
import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N):
    # Calculate total number of iterations
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Check if this thread should process an iteration
    if pid >= num_iterations:
        return
    
    # Calculate i for this iteration
    i = (n1 - 1) + pid * n3
    
    # Calculate k for this iteration: k = j * (pid + 1) where j = 1
    k = pid + 1
    
    # Calculate indices
    b_idx = N - k
    
    # Load values
    a_val = tl.load(a_ptr + i)
    b_val = tl.load(b_ptr + b_idx)
    
    # Compute and store result
    result = a_val + b_val
    tl.store(a_ptr + i, result)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate total number of iterations
    if n1 - 1 >= N:
        return  # No iterations to process
    
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    # Launch kernel
    grid = (num_iterations,)
    s122_kernel[grid](a, b, n1, n3, N)