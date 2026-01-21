import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the original loop
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get block and thread indices
    block_idx = tl.program_id(0)
    
    # Calculate the range of iterations this block will handle
    block_start = block_idx * BLOCK_SIZE
    
    if block_start >= num_iterations:
        return
    
    block_offsets = tl.arange(0, BLOCK_SIZE)
    iteration_indices = block_start + block_offsets
    mask = iteration_indices < num_iterations
    
    # For each valid iteration, compute i, k, and perform the update
    where_mask = iteration_indices < num_iterations
    
    # Compute i values: i = (n1-1) + iteration_index * n3
    i_values = (n1 - 1) + iteration_indices * n3
    
    # Compute k values: k = 1 + iteration_index (since j=1 and k starts at 0, increments by j each time)
    k_values = 1 + iteration_indices
    
    # Compute b indices: LEN_1D - k = N - k_values
    b_indices = N - k_values
    
    # Load values with masking
    a_mask = where_mask & (i_values < N)
    b_mask = where_mask & (b_indices >= 0) & (b_indices < N)
    
    # Load a values
    a_vals = tl.load(a_ptr + i_values, mask=a_mask, other=0.0)
    
    # Load b values
    b_vals = tl.load(b_ptr + b_indices, mask=b_mask, other=0.0)
    
    # Compute update
    updated_vals = a_vals + b_vals
    
    # Store back to a
    tl.store(a_ptr + i_values, updated_vals, mask=a_mask)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the number of iterations in the original loop
    if n1 - 1 >= N:
        return  # No iterations
    
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s122_kernel[grid](
        a, b, n1, n3, N,
        BLOCK_SIZE=BLOCK_SIZE
    )