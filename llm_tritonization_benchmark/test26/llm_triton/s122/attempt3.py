import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate total number of iterations in the original loop
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process iterations in blocks
    for block_start in range(0, num_iterations, BLOCK_SIZE):
        # Calculate iteration indices for this block
        iter_indices = block_start + offsets
        iter_mask = iter_indices < num_iterations
        
        # Calculate i values: i = (n1-1) + iter_index * n3
        i_values = (n1 - 1) + iter_indices * n3
        
        # Calculate k values: k starts at 0, then k += j for each iteration
        # Since j=1, k = iteration_count = iter_indices + 1
        k_values = iter_indices + 1
        
        # Calculate b indices: LEN_1D - k = N - k_values
        b_indices = N - k_values
        
        # Create masks for valid accesses
        i_mask = (i_values >= 0) & (i_values < N)
        b_mask = (b_indices >= 0) & (b_indices < N)
        valid_mask = iter_mask & i_mask & b_mask
        
        # Load values
        a_vals = tl.load(a_ptr + i_values, mask=valid_mask, other=0.0)
        b_vals = tl.load(b_ptr + b_indices, mask=valid_mask, other=0.0)
        
        # Compute and store
        result = a_vals + b_vals
        tl.store(a_ptr + i_values, result, mask=valid_mask)

def s122_triton(a, b):
    N = a.shape[0]
    
    # Default values commonly used in TSVC benchmarks
    n1 = 1
    n3 = 2
    
    BLOCK_SIZE = 256
    
    # Launch kernel with single program
    grid = (1,)
    s122_kernel[grid](a, b, n1, n3, N, BLOCK_SIZE)