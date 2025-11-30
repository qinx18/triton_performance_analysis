import torch
import triton
import triton.language as tl

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Calculate starting offset for this block (reverse order)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Reverse the offsets to process in descending order
    reversed_offsets = n_elements - 1 - offsets
    
    # Create mask for valid elements
    mask = (offsets < n_elements) & (reversed_offsets >= 0)
    
    # Load data
    b_vals = tl.load(b_ptr + reversed_offsets, mask=mask)
    
    # Compute: a[i] = b[i] + 1.0
    result = b_vals + 1.0
    
    # Store result
    tl.store(a_ptr + reversed_offsets, result, mask=mask)

def s1112_triton(a, b):
    n_elements = a.numel()
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s1112_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a