import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate which iteration this block handles
    block_id = tl.program_id(0)
    
    # Calculate the starting index for this block
    # Each block processes BLOCK_SIZE iterations of the strided loop
    block_start = block_id * BLOCK_SIZE
    
    # Generate iteration indices within this block
    iteration_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual array indices: i = (n1-1) + iteration * n3
    array_indices = (n1 - 1) + (block_start + iteration_offsets) * n3
    
    # Create mask for valid indices
    valid_iterations = block_start + iteration_offsets < ((N - (n1 - 1) + n3 - 1) // n3)
    valid_indices = array_indices < N
    mask = valid_iterations & valid_indices
    
    # Load values
    a_vals = tl.load(a_ptr + array_indices, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + array_indices, mask=mask, other=0.0)
    
    # Compute a[i] += b[i]
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + array_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate total number of loop iterations
    if n3 <= 0:
        return
    
    total_iterations = (N - (n1 - 1) + n3 - 1) // n3
    
    if total_iterations <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_iterations, BLOCK_SIZE),)
    
    s172_kernel[grid](
        a, b, n1, n3, N,
        BLOCK_SIZE=BLOCK_SIZE
    )