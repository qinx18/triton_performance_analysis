import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the original loop
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate which iteration this block handles
    pid = tl.program_id(0)
    
    # Process multiple iterations per block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    iteration_ids = block_start + offsets
    
    # Mask for valid iterations
    mask = iteration_ids < num_iterations
    
    # Calculate the actual i values for each iteration
    i_values = (n1 - 1) + iteration_ids * n3
    
    # Calculate k values: k starts at 0 and increments by j=1 each iteration
    # So k = iteration_id * j = iteration_id * 1 = iteration_id
    k_values = iteration_ids + 1  # k starts at 0, then becomes 1, 2, 3, ...
    
    # Calculate b indices: LEN_1D - k = N - k
    b_indices = N - k_values
    
    # Load from arrays with proper masking
    a_vals = tl.load(a_ptr + i_values, mask=mask)
    b_vals = tl.load(b_ptr + b_indices, mask=mask)
    
    # Perform computation
    result = a_vals + b_vals
    
    # Store result back to a
    tl.store(a_ptr + i_values, result, mask=mask)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the number of iterations in the original loop
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    # Use a reasonable block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    # Launch kernel
    s122_kernel[grid](a, b, n1, n3, N, BLOCK_SIZE=BLOCK_SIZE)