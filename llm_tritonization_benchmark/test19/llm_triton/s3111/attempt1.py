import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    total_sum = 0.0
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Apply condition: a[i] > 0
        condition = a_vals > 0.0
        conditional_vals = tl.where(condition, a_vals, 0.0)
        
        block_sum = tl.sum(conditional_vals, axis=0)
        total_sum += block_sum
    
    # Store result
    tl.store(result_ptr, total_sum)

def s3111_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for the result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread block since we need global reduction
    grid = (1,)
    s3111_kernel[grid](
        a, result, n, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()