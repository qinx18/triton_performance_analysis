import torch
import triton
import triton.language as tl

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize minimum with first element
    first_val = tl.load(a_ptr)
    current_min = first_val
    
    # Process remaining elements in blocks
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        block_min = tl.min(vals, axis=0)
        current_min = tl.minimum(current_min, block_min)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, current_min)

def s316_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    grid = (1,)
    s316_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()