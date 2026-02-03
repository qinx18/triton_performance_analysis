import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the original loop
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get block of iterations to process
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    iter_indices = block_start + offsets
    mask = iter_indices < num_iterations
    
    # Calculate i values: i = n1-1 + iter_idx * n3
    i_values = (n1 - 1) + iter_indices * n3
    
    # Calculate k values: k = 1 + iter_idx (since j=1 and k += j for each iteration)
    k_values = 1 + iter_indices
    
    # Calculate b indices: LEN_1D - k = N - k_values
    b_indices = N - k_values
    
    # Load values
    a_vals = tl.load(a_ptr + i_values, mask=mask)
    b_vals = tl.load(b_ptr + b_indices, mask=mask)
    
    # Compute and store
    result = a_vals + b_vals
    tl.store(a_ptr + i_values, result, mask=mask)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the number of iterations
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