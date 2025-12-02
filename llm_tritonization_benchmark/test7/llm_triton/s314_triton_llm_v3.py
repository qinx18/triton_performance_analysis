import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the maximum value in array a
    # Each block processes BLOCK_SIZE elements and finds local max
    # Then we need a second pass to find global max across blocks
    
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load values from array a
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=float('-inf'))
    
    # Find maximum in this block
    block_max = tl.max(vals, axis=0)
    
    # Store block maximum
    tl.store(result_ptr + block_id, block_max)

@triton.jit  
def s314_final_kernel(partial_results_ptr, final_result_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    # Find maximum across all block results
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    # Load partial results
    vals = tl.load(partial_results_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find global maximum
    global_max = tl.max(vals, axis=0)
    
    # Store final result
    tl.store(final_result_ptr, global_max)

def s314_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary storage for partial results
    partial_results = torch.empty(num_blocks, dtype=a.dtype, device=a.device)
    final_result = torch.empty(1, dtype=a.dtype, device=a.device)
    
    # First pass: find maximum in each block
    grid = (num_blocks,)
    s314_kernel[grid](a, partial_results, n_elements, BLOCK_SIZE)
    
    # Second pass: find maximum across block results
    if num_blocks > 1:
        final_grid = (1,)
        final_block_size = min(1024, triton.next_power_of_2(num_blocks))
        s314_final_kernel[final_grid](partial_results, final_result, num_blocks, final_block_size)
    else:
        final_result = partial_results
    
    return final_result.item()