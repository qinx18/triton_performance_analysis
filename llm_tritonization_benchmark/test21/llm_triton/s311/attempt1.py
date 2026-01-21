import triton
import triton.language as tl
import torch

@triton.jit
def s311_kernel(
    a_ptr,
    result_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Single block handles the entire reduction
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize sum
    local_sum = 0.0
    
    # Process array in chunks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        local_sum += tl.sum(vals)
    
    # Store result (only first thread writes)
    if tl.program_id(0) == 0:
        tl.store(result_ptr, local_sum)

def s311_triton(a):
    N = a.shape[0]
    
    # Result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 1024
    
    # Use single block for reduction
    grid = (1,)
    
    s311_kernel[grid](
        a,
        result,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result.item()