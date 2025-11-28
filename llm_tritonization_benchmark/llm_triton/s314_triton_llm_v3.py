import triton
import triton.language as tl
import torch

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each block computes a partial maximum
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find maximum within this block
    block_max = tl.max(a_vals, axis=0)
    
    # Store partial result
    tl.store(result_ptr + tl.program_id(0), block_max)

def s314_triton(a):
    n_elements = a.shape[0]
    
    # First phase: compute partial maxima
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial results
    partial_results = torch.empty(num_blocks, device=a.device, dtype=a.dtype)
    
    # Launch kernel for first reduction phase
    s314_kernel[(num_blocks,)](
        a, partial_results, n_elements, BLOCK_SIZE
    )
    
    # Second phase: find maximum among partial results
    if num_blocks > 1:
        # For small number of blocks, use CPU reduction
        if num_blocks <= 1024:
            result = torch.max(partial_results)
        else:
            # Recursively apply reduction for very large arrays
            result = s314_triton(partial_results)
    else:
        result = partial_results[0]
    
    return result