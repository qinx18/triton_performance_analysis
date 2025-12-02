import torch
import triton
import triton.language as tl

@triton.jit
def s113_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the scalar value a[0] once
    a0 = tl.load(a_ptr)
    
    # Calculate the starting offset for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the block (starting from 1, not 0)
    offsets = block_start + tl.arange(0, BLOCK_SIZE) + 1
    
    # Create mask to ensure we don't go out of bounds and skip index 0
    mask = offsets < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store the result
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    n_elements = a.shape[0]
    
    # We start from index 1, so we need to process n_elements - 1 elements
    elements_to_process = n_elements - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(elements_to_process, BLOCK_SIZE),)
    
    s113_kernel[grid](a, b, n_elements, BLOCK_SIZE)