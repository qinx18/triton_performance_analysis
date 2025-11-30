import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE + 1  # Start from index 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load a[0] - broadcast to all threads
    a_0 = tl.load(a_ptr)
    
    # Load b[i] for this block
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a_0 + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s113_triton(a, b):
    n_elements = a.shape[0]
    
    # Only process elements from index 1 to n_elements-1
    if n_elements <= 1:
        return
    
    BLOCK_SIZE = 256
    # Calculate grid size for elements 1 to n_elements-1
    grid = ((n_elements - 1) + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    s113_kernel[(grid,)](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )