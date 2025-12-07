import torch
import triton
import triton.language as tl

@triton.jit
def s291_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    
    # This kernel processes elements sequentially to preserve dependencies
    # Each block handles BLOCK_SIZE consecutive elements
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Process each element in the block sequentially
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        
        # Check bounds
        valid = idx < n_elements
        
        # Get im1 value (previous index, wrapping around)
        im1_idx = tl.where(idx == 0, n_elements - 1, idx - 1)
        
        # Load values conditionally
        b_i = tl.load(b_ptr + idx, mask=valid, other=0.0)
        b_im1 = tl.load(b_ptr + im1_idx, mask=valid, other=0.0)
        
        # Compute and store
        result = (b_i + b_im1) * 0.5
        tl.store(a_ptr + idx, result, mask=valid)

def s291_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s291_kernel[grid](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a