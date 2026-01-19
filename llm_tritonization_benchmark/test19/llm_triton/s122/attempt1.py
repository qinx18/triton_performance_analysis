import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the original loop
    # Original loop: for (int i = n1-1; i < N; i += n3)
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3  # Ceiling division
    
    # Get block of iterations to process
    block_start = tl.program_id(0) * BLOCK_SIZE
    iteration_offsets = tl.arange(0, BLOCK_SIZE)
    iteration_ids = block_start + iteration_offsets
    iteration_mask = iteration_ids < num_iterations
    
    # Calculate i values: i = (n1-1) + iteration_id * n3
    i_values = (n1 - 1) + iteration_ids * n3
    
    # Calculate k values: k = 1 + iteration_id (since j=1 and k += j each iteration)
    k_values = 1 + iteration_ids
    
    # Calculate b indices: LEN_1D - k = N - k_values
    b_indices = N - k_values
    
    # Create masks for valid accesses
    valid_i = (i_values >= 0) & (i_values < N) & iteration_mask
    valid_b = (b_indices >= 0) & (b_indices < N) & iteration_mask
    
    # Load values
    a_vals = tl.load(a_ptr + i_values, mask=valid_i, other=0.0)
    b_vals = tl.load(b_ptr + b_indices, mask=valid_b, other=0.0)
    
    # Compute and store
    result = a_vals + b_vals
    tl.store(a_ptr + i_values, result, mask=valid_i & valid_b)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate number of iterations
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