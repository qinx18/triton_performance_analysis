import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate number of iterations in the loop
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get block start
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = block_start + offsets
    mask = block_offsets < num_iterations
    
    # Calculate i values for all elements in the block
    iteration_indices = block_offsets
    i_values = (n1 - 1) + iteration_indices * n3
    
    # Calculate k values: k starts at 0, increments by j=1 each iteration
    k_values = iteration_indices + 1
    
    # Calculate b indices
    b_indices = LEN_1D - k_values
    
    # Load values with masking
    a_vals = tl.load(a_ptr + i_values, mask=mask)
    b_vals = tl.load(b_ptr + b_indices, mask=mask)
    
    # Compute new values
    new_vals = a_vals + b_vals
    
    # Store results with masking
    tl.store(a_ptr + i_values, new_vals, mask=mask)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of iterations in the loop
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s122_kernel[grid](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )