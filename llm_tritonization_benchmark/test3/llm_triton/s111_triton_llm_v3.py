import triton
import triton.language as tl
import torch

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle edge cases
    mask = offsets < n_elements
    
    # Process elements in pairs (i = 1, 3, 5, ...)
    # Each thread handles one odd index
    i = 1 + offsets * 2
    
    # Mask for valid odd indices
    valid_mask = (i < n_elements) & mask
    
    # Load a[i-1] and b[i]
    a_prev = tl.load(a_ptr + i - 1, mask=valid_mask, other=0.0)
    b_val = tl.load(b_ptr + i, mask=valid_mask, other=0.0)
    
    # Compute a[i] = a[i-1] + b[i]
    result = a_prev + b_val
    
    # Store result back to a[i]
    tl.store(a_ptr + i, result, mask=valid_mask)

def s111_triton(a, b):
    n_elements = (a.shape[0] - 1) // 2  # Number of odd indices to process
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s111_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a