import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start and create offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data and compute absolute values
    vals = tl.load(a_ptr + offsets, mask=mask, other=-1e9)
    abs_vals = tl.abs(vals)
    
    # Find maximum in this block
    block_max = tl.max(abs_vals)
    
    # Store result
    tl.store(result_ptr + pid, block_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    
    # Handle empty array
    if n_elements == 0:
        return 0.0
    
    # Create output tensor for partial results
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    partial_results = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    
    s3113_kernel[grid](
        a, partial_results, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Find final maximum
    return torch.max(partial_results).item()