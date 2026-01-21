import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < N
    
    # Load values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Apply condition: only sum if vals > 0
    condition_mask = vals > 0.0
    filtered_vals = tl.where(condition_mask & mask, vals, 0.0)
    
    # Sum the filtered values in this block
    block_sum = tl.sum(filtered_vals)
    
    # Store partial result
    tl.store(result_ptr + pid, block_sum)

def s3111_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Create result tensor for partial sums
    partial_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s3111_kernel[grid](a, partial_results, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Sum all partial results
    final_sum = torch.sum(partial_results)
    
    return final_sum.item()