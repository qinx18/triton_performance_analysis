import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize max with absolute value of first element
    if offsets[0] == 0:
        first_val = tl.load(a_ptr)
        current_max = tl.abs(first_val)
    else:
        current_max = 0.0
    
    # Process array in blocks to find maximum absolute value
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        block_max = tl.max(abs_vals, axis=0)
        
        current_max = tl.maximum(current_max, block_max)
    
    # Store result
    tl.store(result_ptr, current_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    
    # Launch kernel with single block to handle reduction
    s3113_kernel[(1,)](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()