import torch
import triton
import triton.language as tl

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, len_1d, BLOCK_SIZE: tl.constexpr):
    # Calculate starting index for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Generate indices for this block
    block_indices = tl.arange(0, BLOCK_SIZE)
    
    # Calculate the strided indices: start from (n1-1), step by n3
    # Map block indices to actual array indices
    actual_indices = (n1 - 1) + (block_start + block_indices) * n3
    
    # Create mask for valid indices
    mask = actual_indices < len_1d
    
    # Load data with masking
    a_vals = tl.load(a_ptr + actual_indices, mask=mask)
    b_vals = tl.load(b_ptr + actual_indices, mask=mask)
    
    # Perform computation
    result = a_vals + b_vals
    
    # Store result with masking
    tl.store(a_ptr + actual_indices, result, mask=mask)

def s172_triton(a, b, n1, n3):
    len_1d = a.shape[0]
    
    # Calculate total number of elements to process
    if n3 <= 0:
        return
    
    total_elements = (len_1d - (n1 - 1) + n3 - 1) // n3
    if total_elements <= 0:
        return
    
    # Use block size that's efficient for the GPU
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    s172_kernel[(num_blocks,)](
        a, b, n1, n3, len_1d,
        BLOCK_SIZE=BLOCK_SIZE
    )