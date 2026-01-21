import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    
    # Handle empty blocks by using -infinity as identity
    masked_vals = tl.where(mask, abs_vals, float('-inf'))
    block_max = tl.max(masked_vals)
    
    tl.store(result_ptr + pid, block_max)

@triton.jit
def s3113_final_kernel(partial_results_ptr, result_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    vals = tl.load(partial_results_ptr + offsets, mask=mask, other=float('-inf'))
    global_max = tl.max(vals)
    
    tl.store(result_ptr, global_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    
    if n_elements == 0:
        return 0.0
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    partial_results = torch.full((max(num_blocks, BLOCK_SIZE),), float('-inf'), dtype=a.dtype, device=a.device)
    
    grid = (num_blocks,)
    s3113_kernel[grid](a, partial_results, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s3113_final_kernel[grid](partial_results, result, num_blocks, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()