import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate the starting index for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate indices based on the strided pattern: i = n1-1, n1-1+n3, n1-1+2*n3, ...
    # Each thread handles one element in the strided sequence
    actual_indices = (n1 - 1) + (block_start + offsets) * n3
    
    # Create mask for valid indices
    mask = (actual_indices >= (n1 - 1)) & (actual_indices < N)
    
    # Load values
    a_vals = tl.load(a_ptr + actual_indices, mask=mask)
    b_vals = tl.load(b_ptr + actual_indices, mask=mask)
    
    # Compute a[i] += b[i]
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    
    # Calculate the number of elements that will be processed
    # Elements are at indices: n1-1, n1-1+n3, n1-1+2*n3, ..., up to N-1
    if n3 <= 0 or n1 - 1 >= N:
        return  # No elements to process
    
    # Calculate the maximum number of steps in the strided pattern
    max_steps = (N - (n1 - 1) + n3 - 1) // n3
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(max_steps, BLOCK_SIZE),)
    
    s172_kernel[grid](a, b, n1, n3, N, BLOCK_SIZE=BLOCK_SIZE)