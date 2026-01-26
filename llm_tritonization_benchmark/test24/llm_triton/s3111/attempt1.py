import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get block offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize block sum
    block_sum = 0.0
    
    # Process all blocks
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        # Load values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Apply condition: if a[i] > 0.0
        condition = vals > 0.0
        conditional_vals = tl.where(condition, vals, 0.0)
        
        # Add to block sum
        block_sum += tl.sum(conditional_vals, axis=0)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, block_sum)

def s3111_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block
    grid = (1,)
    s3111_kernel[grid](a, result, N, BLOCK_SIZE)
    
    return result.item()