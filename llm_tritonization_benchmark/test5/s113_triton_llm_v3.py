import triton
import triton.language as tl
import torch

@triton.jit
def s113_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE + 1  # Start from index 1
    
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Create mask for valid elements (i < n_elements and i >= 1)
    mask = (current_offsets < n_elements) & (current_offsets >= 1)
    
    # Load a[0] (scalar broadcast)
    a0 = tl.load(a_ptr)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    
    # Compute a[i] = a[0] + b[i]
    result = a0 + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s113_triton(a, b):
    n_elements = a.shape[0]
    
    # Block size for efficient memory access
    BLOCK_SIZE = 256
    
    # Calculate grid size (exclude index 0 since we start from 1)
    grid_size = triton.cdiv(n_elements - 1, BLOCK_SIZE)
    
    # Launch kernel
    s113_kernel[(grid_size,)](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )