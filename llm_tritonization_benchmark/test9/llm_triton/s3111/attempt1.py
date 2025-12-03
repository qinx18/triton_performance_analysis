import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize local sum
    local_sum = 0.0
    
    # Process elements in this block
    for start in range(block_start, n_elements, tl.num_programs(0) * BLOCK_SIZE):
        current_offsets = start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Apply condition and sum
        condition = a_vals > 0.0
        conditional_vals = tl.where(condition, a_vals, 0.0)
        local_sum += tl.sum(conditional_vals)
    
    # Store result
    tl.store(result_ptr + pid, local_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s3111_kernel[(num_blocks,)](
        a, partial_sums, n_elements, BLOCK_SIZE
    )
    
    # Sum partial results
    total_sum = torch.sum(partial_sums)
    
    return total_sum