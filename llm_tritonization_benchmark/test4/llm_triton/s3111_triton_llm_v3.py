import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum for this block
    block_sum = 0.0
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load data with masking
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Apply condition: if a[i] > 0
        condition = a_vals > 0.0
        
        # Conditional sum: only add values where condition is true
        masked_vals = tl.where(condition, a_vals, 0.0)
        block_sum += tl.sum(masked_vals)
    
    # Store the sum (only thread 0 in the block writes the result)
    if tl.program_id(0) == 0:
        tl.store(tl.program_id(0) * 1 + 0, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create output tensor for sum
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block to handle the global sum
    grid = (1,)
    
    # Use a more efficient approach with reduction
    @triton.jit
    def reduction_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        
        total_sum = 0.0
        
        # Process entire array in chunks
        for block_start in range(0, n_elements, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < n_elements
            
            # Load values
            a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
            
            # Apply condition and sum
            condition = a_vals > 0.0
            masked_vals = tl.where(condition, a_vals, 0.0)
            total_sum += tl.sum(masked_vals)
        
        # Store final result
        tl.store(output_ptr, total_sum)
    
    reduction_kernel[grid](
        a, output, n_elements, BLOCK_SIZE
    )
    
    return output.item()