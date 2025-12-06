import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum accumulator
    sum_val = 0.0
    
    # Process elements in blocks
    for start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Apply condition and accumulate
        condition_mask = a_vals > 0.0
        masked_vals = tl.where(condition_mask, a_vals, 0.0)
        sum_val += tl.sum(masked_vals)
    
    # Store result (only first thread writes the final sum)
    if pid == 0:
        tl.store(result_ptr, sum_val)

def s3111_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Use single program to handle the reduction
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s3111_kernel[grid](
        a, result, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()