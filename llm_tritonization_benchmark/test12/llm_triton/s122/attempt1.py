import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate total number of iterations in the original loop
    total_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate which iteration this block handles
    pid = tl.program_id(0)
    
    # Each block processes BLOCK_SIZE iterations
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    iteration_indices = block_start + offsets
    
    # Mask for valid iterations
    mask = iteration_indices < total_iterations
    
    # Calculate actual i values: i = (n1-1) + iteration_idx * n3
    i_values = (n1 - 1) + iteration_indices * n3
    
    # Calculate k values: k = 1 + iteration_idx (since j=1 and k starts at 0)
    k_values = 1 + iteration_indices
    
    # Calculate b indices: LEN_1D - k
    b_indices = LEN_1D - k_values
    
    # Ensure indices are within bounds
    i_mask = (i_values < LEN_1D) & (i_values >= 0) & mask
    b_mask = (b_indices < LEN_1D) & (b_indices >= 0) & mask
    combined_mask = i_mask & b_mask
    
    # Load values from b array
    b_values = tl.load(b_ptr + b_indices, mask=combined_mask, other=0.0)
    
    # Load current values from a array
    a_values = tl.load(a_ptr + i_values, mask=combined_mask, other=0.0)
    
    # Compute new values
    new_a_values = a_values + b_values
    
    # Store back to a array
    tl.store(a_ptr + i_values, new_a_values, mask=combined_mask)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate total number of iterations
    if n3 <= 0 or n1 - 1 >= LEN_1D:
        return  # No iterations to process
        
    total_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if total_iterations <= 0:
        return
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(total_iterations, BLOCK_SIZE)
    
    # Launch kernel
    s122_kernel[(grid_size,)](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )