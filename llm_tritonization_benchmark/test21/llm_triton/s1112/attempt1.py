import triton
import triton.language as tl
import torch

@triton.jit
def s1112_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate starting position for this block (reverse order)
    block_id = tl.program_id(0)
    block_start = n_elements - (block_id + 1) * BLOCK_SIZE
    
    # Generate offsets within the block
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices
    mask = (indices >= 0) & (indices < n_elements)
    
    # Load data from b array
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] = b[i] + 1.0
    a_vals = b_vals + 1.0
    
    # Store results to a array
    tl.store(a_ptr + indices, a_vals, mask=mask)

def s1112_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size (number of blocks needed)
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (grid_size,)
    
    # Launch kernel
    s1112_kernel[grid](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )