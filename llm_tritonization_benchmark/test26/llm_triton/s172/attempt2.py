import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate total number of iterations in the original loop
    num_iterations = (N - n1 + n3) // n3
    
    # Block processing
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block (iteration numbers)
    offsets = tl.arange(0, BLOCK_SIZE)
    iteration_nums = block_start + offsets
    
    # Convert iteration numbers to actual array indices
    actual_indices = n1 + iteration_nums * n3
    
    # Mask for valid iterations
    mask = (iteration_nums < num_iterations) & (actual_indices < N)
    
    # Load values
    a_vals = tl.load(a_ptr + actual_indices, mask=mask)
    b_vals = tl.load(b_ptr + actual_indices, mask=mask)
    
    # Compute and store
    result = a_vals + b_vals
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s172_triton(a, b):
    N = a.shape[0]
    n1 = 5  # Default values
    n3 = 2
    
    # Calculate number of iterations
    num_iterations = (N - n1 + n3) // n3
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s172_kernel[grid](a, b, n1, n3, N, BLOCK_SIZE)