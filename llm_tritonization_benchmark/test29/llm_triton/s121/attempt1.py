import triton
import triton.language as tl
import torch

@triton.jit
def s121_kernel(
    a_ptr, b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Compute indices for this block
    i_offsets = block_start + offsets
    j_offsets = i_offsets + 1
    
    # Create masks
    i_mask = i_offsets < n_elements
    j_mask = j_offsets < (n_elements + 1)  # j can go up to n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + i_offsets, mask=i_mask, other=0.0)
    a_j_vals = tl.load(a_ptr + j_offsets, mask=j_mask, other=0.0)
    
    # Compute result
    result = a_j_vals + b_vals
    
    # Store result
    tl.store(a_ptr + i_offsets, result, mask=i_mask)

def s121_triton(a, b):
    # Get array size
    N = a.shape[0]
    n_elements = N - 1  # Loop goes from 0 to N-2 (N-1 elements)
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s121_kernel[(grid_size,)](
        a, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a