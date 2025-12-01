import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, LEN_1D, BLOCK_SIZE: tl.constexpr):
    # Calculate total number of iterations in the strided loop
    total_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block (in iteration space)
    offsets = tl.arange(0, BLOCK_SIZE)
    iter_offsets = block_start + offsets
    
    # Mask for valid iterations
    mask = iter_offsets < total_iterations
    
    # Convert iteration indices to actual array indices
    array_indices = (n1 - 1) + iter_offsets * n3
    
    # Load values from arrays a and b
    a_vals = tl.load(a_ptr + array_indices, mask=mask)
    b_vals = tl.load(b_ptr + array_indices, mask=mask)
    
    # Perform the computation: a[i] += b[i]
    result = a_vals + b_vals
    
    # Store back to array a
    tl.store(a_ptr + array_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    LEN_1D = a.shape[0]
    
    # Calculate total number of iterations
    total_iterations = (LEN_1D - (n1 - 1) + n3 - 1) // n3
    
    if total_iterations <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_iterations, BLOCK_SIZE),)
    
    s172_kernel[grid](
        a, b, n1, n3, LEN_1D,
        BLOCK_SIZE=BLOCK_SIZE
    )