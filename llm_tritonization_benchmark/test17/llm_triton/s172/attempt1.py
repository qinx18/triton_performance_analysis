import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n_elements, n1, n3, BLOCK_SIZE: tl.constexpr):
    # Calculate how many indices we'll process
    num_indices = (n_elements - (n1 - 1) + n3 - 1) // n3
    
    # Get program ID and calculate block boundaries
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    block_end = block_start + BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    idx_offsets = block_start + offsets
    
    # Mask for valid indices in this block
    mask = idx_offsets < num_indices
    
    # Convert sequential indices to actual array indices: i = n1-1 + idx*n3
    array_indices = (n1 - 1) + idx_offsets * n3
    
    # Load values
    a_vals = tl.load(a_ptr + array_indices, mask=mask)
    b_vals = tl.load(b_ptr + array_indices, mask=mask)
    
    # Compute
    result = a_vals + b_vals
    
    # Store back
    tl.store(a_ptr + array_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    n_elements = a.shape[0]
    
    # Calculate how many indices we'll actually process
    if n3 <= 0 or n1 - 1 >= n_elements:
        return
    
    num_indices = (n_elements - (n1 - 1) + n3 - 1) // n3
    
    if num_indices <= 0:
        return
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(num_indices, BLOCK_SIZE),)
    
    s172_kernel[grid](
        a, b, n_elements, n1, n3,
        BLOCK_SIZE=BLOCK_SIZE
    )