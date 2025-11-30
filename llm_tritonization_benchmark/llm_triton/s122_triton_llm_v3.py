import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the original loop
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which iteration this thread block handles
    iter_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid iterations
    mask = iter_idx < num_iterations
    
    # Calculate i values: i = n1-1 + iter_idx * n3
    i_vals = (n1 - 1) + iter_idx * n3
    
    # Calculate k values: k = 1 + iter_idx (since j=1 and k += j each iteration)
    k_vals = 1 + iter_idx
    
    # Calculate b indices: LEN_1D - k
    b_indices = LEN_1D - k_vals
    
    # Load values with masking
    a_vals = tl.load(a_ptr + i_vals, mask=mask)
    b_vals = tl.load(b_ptr + b_indices, mask=mask)
    
    # Perform computation
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + i_vals, result, mask=mask)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate the number of iterations
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    # Launch kernel
    s122_kernel[grid_size](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )