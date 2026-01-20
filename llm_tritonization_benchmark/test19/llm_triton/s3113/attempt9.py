import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize with negative infinity to handle max reduction properly
    block_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
        abs_vals = tl.abs(vals)
        
        # Update block_max element-wise
        block_max = tl.maximum(block_max, abs_vals)
    
    # Reduce across the block to get final maximum
    final_max = tl.max(block_max)
    
    # Store result
    tl.store(result_ptr, final_max)

def s3113_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    s3113_kernel[(1,)](a, result, n, BLOCK_SIZE=BLOCK_SIZE)
    return result.item()