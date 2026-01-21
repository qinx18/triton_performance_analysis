import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    
    block_max = tl.max(abs_vals, axis=0)
    
    tl.store(result_ptr + program_id, block_max)

@triton.jit
def s3113_final_kernel(partial_results_ptr, result_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    vals = tl.load(partial_results_ptr + offsets, mask=mask, other=0.0)
    
    global_max = tl.max(vals, axis=0)
    
    tl.store(result_ptr, global_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    if num_blocks == 1:
        result = torch.zeros(1, dtype=a.dtype, device=a.device)
        s3113_kernel[(1,)](a, result, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        return result.item()
    
    partial_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    grid = (num_blocks,)
    s3113_kernel[grid](a, partial_results, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    final_block_size = max(64, triton.next_power_of_2(num_blocks))
    s3113_final_kernel[(1,)](partial_results, result, num_blocks, BLOCK_SIZE=final_block_size)
    
    return result.item()