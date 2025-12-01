import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate total number of iterations
    total_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate block start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    iter_offsets = block_start + offsets
    
    # Calculate actual array indices: i = n1-1 + iter_idx * n3
    array_indices = (n1 - 1) + iter_offsets * n3
    
    # Create mask for valid iterations and array bounds
    mask = (iter_offsets < total_iterations) & (array_indices < LEN_1D)
    
    # Load values
    a_vals = tl.load(a_ptr + array_indices, mask=mask)
    b_vals = tl.load(b_ptr + array_indices, mask=mask)
    
    # Compute a[i] += b[i]
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + array_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate total number of iterations in the strided loop
    total_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if total_iterations <= 0:
        return
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(total_iterations, BLOCK_SIZE),)
    
    # Launch kernel
    s172_kernel[grid](
        a, b, n1, n3, LEN_1D, BLOCK_SIZE
    )