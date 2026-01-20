import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate number of iterations in the original loop
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate block boundaries
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    iteration_indices = block_start + offsets
    
    # Mask for valid iterations
    mask = iteration_indices < num_iterations
    
    # Calculate i values: i = (n1-1) + iteration_index * n3
    i_values = (n1 - 1) + iteration_indices * n3
    
    # Calculate k values: k = j * (iteration_index + 1), where j = 1
    k_values = iteration_indices + 1
    
    # Calculate b indices: LEN_1D - k = N - k_values
    b_indices = N - k_values
    
    # Load from a and b arrays
    a_values = tl.load(a_ptr + i_values, mask=mask)
    b_values = tl.load(b_ptr + b_indices, mask=mask)
    
    # Perform computation: a[i] += b[N - k]
    result = a_values + b_values
    
    # Store result back to a
    tl.store(a_ptr + i_values, result, mask=mask)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate number of iterations in the original loop
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s122_kernel[grid](
        a, b, n1, n3, N,
        BLOCK_SIZE=BLOCK_SIZE
    )