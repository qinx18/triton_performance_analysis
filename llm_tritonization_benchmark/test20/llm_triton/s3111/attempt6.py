import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Single thread (PID 0) does the entire reduction
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Process array in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    sum_val = 0.0
    
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        # Load values
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Apply condition: only sum if vals > 0
        condition_mask = vals > 0.0
        filtered_vals = tl.where(condition_mask, vals, 0.0)
        
        # Sum the filtered values
        block_sum = tl.sum(filtered_vals)
        sum_val += block_sum
    
    # Store final result
    tl.store(result_ptr, sum_val)

def s3111_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single thread
    grid = (1,)
    s3111_kernel[grid](a, result, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()