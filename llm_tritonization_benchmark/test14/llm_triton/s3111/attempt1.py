import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load block of data
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Apply condition and compute partial sum
    condition_mask = a_vals > 0.0
    conditional_vals = tl.where(condition_mask, a_vals, 0.0)
    
    # Sum the values in this block
    block_sum = tl.sum(conditional_vals)
    
    # Store the partial sum
    tl.store(result_ptr + tl.program_id(0), block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(n_blocks, device=a.device, dtype=a.dtype)
    
    # Launch kernel
    grid = (n_blocks,)
    s3111_kernel[grid](a, partial_sums, n_elements, BLOCK_SIZE)
    
    # Sum all partial results
    sum_result = torch.sum(partial_sums)
    
    return sum_result