import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Find minimum value using parallel reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize local minimum to first element
    mask = offsets < n_elements
    vals = tl.load(a_ptr + offsets, mask=mask, other=float('inf'))
    local_min = tl.min(vals)
    
    # Process remaining elements in blocks
    for block_start in range(BLOCK_SIZE, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        block_min = tl.min(vals)
        local_min = tl.minimum(local_min, block_min)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, local_min)

def s316_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block
    grid = (1,)
    s316_kernel[grid](
        a, result, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()