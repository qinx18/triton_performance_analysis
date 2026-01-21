import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize result with first element
    first_val = tl.load(a_ptr)
    
    # Find minimum across all blocks
    min_val = first_val
    
    # Process remaining elements in blocks
    for block_start in range(BLOCK_SIZE, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        block_min = tl.min(vals, axis=0)
        
        min_val = tl.minimum(min_val, block_min)
    
    # Handle remaining elements in first block (indices 1 to min(BLOCK_SIZE, n-1))
    if BLOCK_SIZE > 1:
        remaining_offsets = 1 + offsets
        mask = remaining_offsets < tl.minimum(BLOCK_SIZE, n)
        
        vals = tl.load(a_ptr + remaining_offsets, mask=mask, other=float('inf'))
        block_min = tl.min(vals, axis=0)
        
        min_val = tl.minimum(min_val, block_min)
    
    # Store result (only first thread stores)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, min_val)

def s316_triton(a):
    n = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single block reduction
    
    s316_kernel[grid](a, result, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()