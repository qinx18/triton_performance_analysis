import torch
import triton
import triton.language as tl

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Initialize with first element
    first_val = tl.load(a_ptr)
    min_val = first_val
    
    # Process remaining elements in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        block_min = tl.min(vals)
        
        # Update global minimum if block minimum is smaller
        min_val = tl.minimum(min_val, block_min)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, min_val)

def s316_triton(a):
    n_elements = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s316_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result[0].item()