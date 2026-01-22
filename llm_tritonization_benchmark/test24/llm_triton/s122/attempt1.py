import triton
import triton.language as tl
import torch

@triton.jit
def s122_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the number of iterations in the original loop
    num_iters = (N - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate which iterations this block handles
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    iter_indices = block_start + offsets
    
    # Mask for valid iterations
    mask = iter_indices < num_iters
    
    # Calculate actual array indices: i = n1-1 + iter_index * n3
    i_indices = (n1 - 1) + iter_indices * n3
    
    # Calculate k values: k = j * (iter_index + 1) where j = 1
    k_values = iter_indices + 1
    
    # Calculate b array indices: LEN_1D - k = N - k_values
    b_indices = N - k_values
    
    # Load values with masking
    a_vals = tl.load(a_ptr + i_indices, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + b_indices, mask=mask, other=0.0)
    
    # Perform computation
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + i_indices, result, mask=mask)

def s122_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the number of iterations
    if n1 - 1 >= N or n3 <= 0:
        return  # No iterations
    
    num_iters = (N - (n1 - 1) + n3 - 1) // n3
    if num_iters <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_iters, BLOCK_SIZE),)
    
    s122_kernel[grid](a, b, n1, n3, N, BLOCK_SIZE=BLOCK_SIZE)