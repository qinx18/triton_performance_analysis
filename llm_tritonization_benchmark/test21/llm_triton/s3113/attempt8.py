import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles one block
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load values with proper masking
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    
    # Find maximum in this block
    block_max = tl.max(abs_vals, axis=0)
    
    # Store block maximum
    tl.store(result_ptr + program_id, block_max)

@triton.jit
def s3113_final_kernel(partial_results_ptr, result_ptr, num_blocks):
    # Find maximum across all partial results
    max_val = tl.load(partial_results_ptr)
    
    for i in range(1, num_blocks):
        val = tl.load(partial_results_ptr + i)
        max_val = tl.maximum(max_val, val)
    
    tl.store(result_ptr, max_val)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create partial results tensor
    partial_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # First pass: compute block-wise maximums
    grid = (num_blocks,)
    s3113_kernel[grid](a, partial_results, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Second pass: find global maximum
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    s3113_final_kernel[(1,)](partial_results, result, num_blocks)
    
    return result.item()