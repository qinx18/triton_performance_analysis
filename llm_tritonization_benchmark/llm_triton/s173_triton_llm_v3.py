import triton
import triton.language as tl
import torch

@triton.jit
def s173_kernel(
    a_ptr, b_ptr,
    n_elements,
    k,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    # Calculate offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    indices = block_start + offsets
    
    # Create mask for valid elements
    mask = indices < n_elements
    
    # Load data from a[i] and b[i]
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store result to a[i+k]
    output_indices = indices + k
    tl.store(a_ptr + output_indices, result, mask=mask)

def s173_triton(a, b):
    # Get problem size
    n_elements = a.shape[0] // 2
    k = a.shape[0] // 2
    
    # Define block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s173_kernel[grid](
        a, b,
        n_elements,
        k,
        BLOCK_SIZE=BLOCK_SIZE,
    )