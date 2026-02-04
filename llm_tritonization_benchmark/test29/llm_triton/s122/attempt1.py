import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the original loop
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get block ID and calculate iteration range for this block
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    iteration_ids = block_start + offsets
    
    # Mask for valid iterations
    mask = iteration_ids < num_iterations
    
    # Calculate actual i values: i = (n1-1) + iteration_id * n3
    i_values = (n1 - 1) + iteration_ids * n3
    
    # Calculate k values: k = 1 + iteration_id (since j=1 and k += j each iteration)
    k_values = 1 + iteration_ids
    
    # Calculate b indices: LEN_1D - k
    b_indices = LEN_1D - k_values
    
    # Load values with masking
    a_vals = tl.load(a_ptr + i_values, mask=mask)
    b_vals = tl.load(b_ptr + b_indices, mask=mask)
    
    # Perform computation
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + i_values, result, mask=mask)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate number of iterations in the loop
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(num_iterations, BLOCK_SIZE)
    
    # Launch kernel
    s122_kernel[(grid_size,)](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )