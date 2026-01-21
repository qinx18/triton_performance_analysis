import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate this block's starting position
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values and compute absolute values
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    
    # Find maximum absolute value in this block
    block_max = tl.max(abs_vals)
    
    # Store the block maximum
    tl.store(result_ptr + pid, block_max)

@triton.jit
def s3113_final_kernel(partial_results_ptr, result_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    global_max = 0.0
    for block_start in range(0, num_blocks, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < num_blocks
        
        vals = tl.load(partial_results_ptr + current_offsets, mask=mask, other=0.0)
        current_max = tl.max(vals)
        global_max = tl.maximum(global_max, current_max)
    
    tl.store(result_ptr, global_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create tensor for partial results
    partial_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch first kernel to compute block maximums
    grid = (num_blocks,)
    s3113_kernel[grid](a, partial_results, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Create final result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch second kernel to find global maximum
    grid = (1,)
    s3113_final_kernel[grid](partial_results, result, num_blocks, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()