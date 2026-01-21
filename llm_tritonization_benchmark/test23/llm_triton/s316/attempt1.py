import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    first_val = tl.load(a_ptr)
    min_val = first_val
    
    # Process remaining elements in blocks
    for block_start in range(BLOCK_SIZE, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        block_min = tl.min(vals, axis=0)
        min_val = tl.minimum(min_val, block_min)
    
    # Handle remaining elements
    remaining_start = ((n - 1) // BLOCK_SIZE) * BLOCK_SIZE + BLOCK_SIZE
    if remaining_start < n:
        remaining_offsets = remaining_start + offsets
        mask = remaining_offsets < n
        vals = tl.load(a_ptr + remaining_offsets, mask=mask, other=float('inf'))
        block_min = tl.min(vals, axis=0)
        min_val = tl.minimum(min_val, block_min)
    
    # Store result
    tl.store(result_ptr, min_val)

def s316_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.empty(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s316_kernel[grid](a, result, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()