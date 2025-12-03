import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate starting position and stride
    start_idx = n1 - 1
    
    # Calculate how many elements we'll process with this stride
    if n3 <= 0:
        return
    
    max_iterations = (n_elements - start_idx + n3 - 1) // n3
    
    for iteration in range(0, max_iterations, BLOCK_SIZE):
        # Calculate indices for this iteration
        iter_offsets = iteration + offsets
        indices = start_idx + iter_offsets * n3
        
        # Create mask for valid indices
        mask = (iter_offsets < max_iterations) & (indices < n_elements) & (indices >= 0)
        
        # Load values
        a_vals = tl.load(a_ptr + indices, mask=mask)
        b_vals = tl.load(b_ptr + indices, mask=mask)
        
        # Compute and store
        result = a_vals + b_vals
        tl.store(a_ptr + indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    BLOCK_SIZE = 256
    n_elements = a.shape[0]
    
    # Launch kernel
    grid = (1,)
    s172_kernel[grid](
        a, b, n1, n3, n_elements, BLOCK_SIZE
    )
    
    return a