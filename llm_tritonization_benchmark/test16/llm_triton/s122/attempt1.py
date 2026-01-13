import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate total number of iterations in the original loop
    if n1 - 1 >= N:
        return
    
    num_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    # Process iterations in blocks
    for block_start in range(0, num_iterations, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        iter_indices = block_start + offsets
        mask = iter_indices < num_iterations
        
        # Calculate i values: i = n1-1 + iter_index * n3
        i_values = (n1 - 1) + iter_indices * n3
        
        # Calculate k values: k = j * (iter_index + 1), where j = 1
        k_values = iter_indices + 1
        
        # Calculate b indices: b[LEN_1D - k] = b[N - k]
        b_indices = N - k_values
        
        # Load values
        a_vals = tl.load(a_ptr + i_values, mask=mask)
        b_vals = tl.load(b_ptr + b_indices, mask=mask)
        
        # Compute and store
        result = a_vals + b_vals
        tl.store(a_ptr + i_values, result, mask=mask)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    BLOCK_SIZE = 256
    
    # Launch kernel with single program
    s122_kernel[(1,)](
        a, b, n1, n3, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a