import triton
import triton.language as tl
import torch

@triton.jit
def s316_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This is a reduction kernel to find minimum value
    pid = tl.program_id(axis=0)
    
    # Load block of data
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values with large value for masked elements
    values = tl.load(a_ptr + offsets, mask=mask, other=float('inf'))
    
    # Reduce within block to find minimum
    block_min = tl.min(values, axis=0)
    
    # Store partial result
    tl.store(result_ptr + pid, block_min)

def s316_triton(a):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary array to store partial results
    partial_results = torch.empty(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel to compute partial minimums
    grid = (num_blocks,)
    s316_kernel[grid](
        a, partial_results, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Final reduction on CPU for partial results
    # This is efficient since partial_results is small
    result = torch.min(partial_results).item()
    
    return result