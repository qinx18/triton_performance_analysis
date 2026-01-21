import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize max_val to negative infinity for proper reduction
    max_val = float('-inf')
    
    # Process array in blocks
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute absolute values and find block maximum
        abs_vals = tl.abs(vals)
        block_max = tl.max(abs_vals, axis=0)
        
        # Update global maximum
        max_val = tl.maximum(max_val, block_max)
    
    # Store result
    tl.store(result_ptr, max_val)

def s3113_triton(a):
    n = a.shape[0]
    
    # Create output tensor for the result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Use a single thread block since we need global reduction
    BLOCK_SIZE = 256
    grid = (1,)
    
    s3113_kernel[grid](
        a, result, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()