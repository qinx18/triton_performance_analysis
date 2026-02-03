import triton
import triton.language as tl
import torch

@triton.jit
def s311_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    local_sum = 0.0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        local_sum += tl.sum(vals, axis=0)
    
    tl.store(result_ptr, local_sum)

def s311_triton(a):
    N = a.shape[0]
    
    # Create result tensor
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 1024
    grid = (1,)
    
    s311_kernel[grid](
        a, result, N, BLOCK_SIZE
    )
    
    return result.item()