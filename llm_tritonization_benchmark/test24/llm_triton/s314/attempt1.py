import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with first element
    first_val = tl.load(a_ptr)
    block_max = first_val
    
    # Process array in blocks
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        current_max = tl.max(vals, axis=0)
        
        # Update global maximum
        block_max = tl.maximum(block_max, current_max)
    
    # Store result
    if tl.program_id(0) == 0:
        tl.store(result_ptr, block_max)

def s314_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Result tensor to store the maximum value
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we need global reduction
    grid = (1,)
    s314_kernel[grid](a, result, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()