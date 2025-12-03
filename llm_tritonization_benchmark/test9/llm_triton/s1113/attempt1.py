import triton
import triton.language as tl
import torch

@triton.jit
def s1113_kernel(a_ptr, a_copy_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load the scalar value from a_copy[LEN_1D/2]
    mid_idx = n_elements // 2
    a_mid = tl.load(a_copy_ptr + mid_idx)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] = a[LEN_1D/2] + b[i]
    result = a_mid + b_vals
    
    # Store results to original array a
    tl.store(a_ptr + offsets, result, mask=mask)

def s1113_triton(a, b):
    n_elements = a.numel()
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Define block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s1113_kernel[grid](
        a, a_copy, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )