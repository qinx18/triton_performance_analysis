import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Initialize with a[0]
    first_val = tl.load(a_ptr)
    current_max = first_val
    
    # Process array in blocks to find maximum
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        block_max = tl.max(vals, axis=0)
        
        # Update current maximum
        current_max = tl.maximum(current_max, block_max)
    
    # Store result (only one thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, current_max)

def s314_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    grid = (1,)
    s314_kernel[grid](a, result, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()