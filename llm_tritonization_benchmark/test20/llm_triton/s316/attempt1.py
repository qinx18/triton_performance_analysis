import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    first_val = tl.load(a_ptr)
    block_min = first_val
    
    # Process remaining elements in blocks
    for block_start in range(BLOCK_SIZE, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        block_min = tl.minimum(block_min, tl.min(vals, axis=0))
    
    # Handle remaining elements if n-1 is not divisible by BLOCK_SIZE
    remaining_start = ((n - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if remaining_start < n - 1:
        remaining_start = max(remaining_start, 1)
        remaining_offsets = remaining_start + offsets
        mask = remaining_offsets < n
        vals = tl.load(a_ptr + remaining_offsets, mask=mask, other=float('inf'))
        block_min = tl.minimum(block_min, tl.min(vals, axis=0))
    
    tl.store(result_ptr, block_min)

def s316_triton(a):
    n = a.shape[0]
    
    if n == 0:
        return torch.tensor(0.0, dtype=a.dtype, device=a.device)
    
    if n == 1:
        return a[0].clone()
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s316_kernel[grid](
        a, result, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result[0]