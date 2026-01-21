import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    sum_val = 0.0
    
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Apply condition: a[i] > 0.0
        condition = vals > 0.0
        
        # Sum only values that meet the condition
        conditional_vals = tl.where(condition, vals, 0.0)
        block_sum = tl.sum(conditional_vals)
        
        sum_val += block_sum
    
    # Store the result
    tl.store(result_ptr, sum_val)

def s3111_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create output tensor for the sum
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we need global reduction
    grid = (1,)
    s3111_kernel[grid](a, result, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()