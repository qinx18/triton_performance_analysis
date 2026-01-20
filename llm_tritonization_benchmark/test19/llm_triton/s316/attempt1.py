import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Single program handles entire reduction
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Initialize minimum with first element
    first_val = tl.load(a_ptr)
    current_min = first_val
    
    # Process remaining elements in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(1, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('inf'))
        block_min = tl.min(vals)
        
        # Update global minimum
        current_min = tl.minimum(current_min, block_min)
    
    # Store final result
    tl.store(result_ptr, current_min)

def s316_triton(a):
    n = a.shape[0]
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s316_kernel[grid](
        a, result, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()