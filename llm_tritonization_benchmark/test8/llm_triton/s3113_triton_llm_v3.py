import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize max with absolute value of first element
    first_val = tl.load(a_ptr)
    current_max = tl.abs(first_val)
    
    # Process array in blocks to find maximum absolute value
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        
        # Find maximum in this block
        block_max = tl.max(abs_vals, axis=0)
        current_max = tl.maximum(current_max, block_max)
    
    # Store result (only first thread stores the final result)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, current_max)

def s3113_triton(a, abs):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Create output tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single block since we need global reduction
    grid = (1,)
    s3113_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()