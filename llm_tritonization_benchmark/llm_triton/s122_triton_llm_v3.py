import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n_elements, n1, n3, BLOCK_SIZE: tl.constexpr):
    # Calculate total number of iterations in the original loop
    total_iters = (n_elements - (n1 - 1) + n3 - 1) // n3
    
    # Process iterations in blocks
    for block_start in range(0, total_iters, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        iter_indices = block_start + offsets
        mask = iter_indices < total_iters
        
        # Calculate corresponding i values: i = n1-1 + iter_idx * n3
        i_values = (n1 - 1) + iter_indices * n3
        
        # Calculate k values: k = 1 + iter_idx (since j=1 and k += j each iteration)
        k_values = 1 + iter_indices
        
        # Calculate b indices: LEN_1D - k
        b_indices = n_elements - k_values
        
        # Load values with masking
        a_vals = tl.load(a_ptr + i_values, mask=mask)
        b_vals = tl.load(b_ptr + b_indices, mask=mask)
        
        # Perform computation
        result = a_vals + b_vals
        
        # Store result
        tl.store(a_ptr + i_values, result, mask=mask)

def s122_triton(a, b, n1, n3):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel
    s122_kernel[(1,)](
        a, b, n_elements, n1, n3, BLOCK_SIZE
    )
    
    return a