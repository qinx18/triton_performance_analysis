import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    block_max = tl.max(abs_vals)
    
    tl.store(result_ptr + pid, block_max)

@triton.jit
def s3113_final_kernel(partial_ptr, result_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    vals = tl.load(partial_ptr + offsets, mask=mask, other=-float('inf'))
    global_max = tl.max(vals)
    
    tl.store(result_ptr, global_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    partial_results = torch.full((num_blocks,), -float('inf'), dtype=a.dtype, device=a.device)
    grid = (num_blocks,)
    s3113_kernel[grid](a, partial_results, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    final_block_size = min(BLOCK_SIZE, num_blocks)
    grid = (1,)
    s3113_final_kernel[grid](partial_results, result, num_blocks, BLOCK_SIZE=final_block_size)
    
    return result.item()