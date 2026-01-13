import triton
import triton.language as tl
import torch

@triton.jit
def s1112_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offset range
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices (reverse order)
    indices = n_elements - 1 - (block_start + offsets)
    
    # Create mask for valid elements
    mask = (block_start + offsets) < n_elements
    
    # Load b values
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] = b[i] + 1.0
    a_vals = b_vals + 1.0
    
    # Store results
    tl.store(a_ptr + indices, a_vals, mask=mask)

def s1112_triton(a, b):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1112_kernel[grid](
        a,
        b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )