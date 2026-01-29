import triton
import triton.language as tl
import torch

@triton.jit
def s311_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute sum reduction across all elements
    total_sum = 0.0
    
    # Process elements in blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        block_sum = tl.sum(vals, axis=0)
        total_sum += block_sum
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, total_sum)

def s311_triton(a):
    n_elements = a.shape[0]
    
    # Allocate result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Launch kernel with single program
    grid = (1,)
    s311_kernel[grid](
        a, result, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()