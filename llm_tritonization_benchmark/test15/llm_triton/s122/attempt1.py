import torch
import triton
import triton.language as tl

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate total number of iterations in the sequential loop
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Each program handles one block of iterations
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    iteration_indices = block_start + offsets
    
    # Mask for valid iterations
    mask = iteration_indices < num_iterations
    
    # Calculate actual i values and k values for valid iterations
    i_values = (n1 - 1) + iteration_indices * n3
    k_values = 1 + iteration_indices  # k starts at 0, then becomes 1, 2, 3, ...
    
    # Calculate indices for array accesses
    a_indices = i_values
    b_indices = LEN_1D - k_values
    
    # Bounds checking
    a_valid = (a_indices >= 0) & (a_indices < LEN_1D)
    b_valid = (b_indices >= 0) & (b_indices < LEN_1D)
    valid_mask = mask & a_valid & b_valid
    
    # Load values
    a_vals = tl.load(a_ptr + a_indices, mask=valid_mask, other=0.0)
    b_vals = tl.load(b_ptr + b_indices, mask=valid_mask, other=0.0)
    
    # Compute and store
    result = a_vals + b_vals
    tl.store(a_ptr + a_indices, result, mask=valid_mask)

def s122_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate total iterations needed
    num_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if num_iterations <= 0:
        return
    
    # Choose block size
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(num_iterations, BLOCK_SIZE)
    
    s122_kernel[(grid_size,)](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )