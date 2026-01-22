import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate total number of elements to process
    # From n1-1 to N-1 with step n3
    start_idx = n1 - 1
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate the starting iteration for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Convert iteration indices to actual array indices
    # Each iteration k corresponds to array index: start_idx + k * n3
    iteration_indices = block_start + offsets
    array_indices = start_idx + iteration_indices * n3
    
    # Create mask for valid indices
    mask = (array_indices < N) & (array_indices >= start_idx)
    
    # Load values
    a_vals = tl.load(a_ptr + array_indices, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + array_indices, mask=mask, other=0.0)
    
    # Compute a[i] += b[i]
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + array_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the number of iterations in the original loop
    start_idx = n1 - 1
    if start_idx >= N or n3 <= 0:
        return  # No valid iterations
    
    # Calculate total number of iterations
    num_iterations = (N - start_idx + n3 - 1) // n3
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iterations, BLOCK_SIZE),)
    
    s172_kernel[grid](a, b, n1, n3, N, BLOCK_SIZE=BLOCK_SIZE)