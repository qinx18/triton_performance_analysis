import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize global max with negative infinity
    global_max = float('-inf')
    
    # Process array in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)
        block_max = tl.max(abs_vals, axis=0)
        
        global_max = tl.maximum(global_max, block_max)
    
    # Store result only from first program
    if tl.program_id(0) == 0:
        tl.store(result_ptr, global_max)

def s3113_triton(a):
    n = a.shape[0]
    
    # Create output tensor for result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program for global reduction
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s3113_kernel[grid](
        a, result, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()