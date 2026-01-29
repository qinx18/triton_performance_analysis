import triton
import triton.language as tl
import torch

@triton.jit
def vsumr_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_sum = 0.0
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        block_sum += tl.sum(vals)
    
    # Store result from first thread
    if tl.program_id(0) == 0:
        tl.store(result_ptr, block_sum)

def vsumr_triton(a):
    n_elements = a.shape[0]
    
    # Output tensor for result
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Launch single block since we need global reduction
    grid = (1,)
    BLOCK_SIZE = 1024
    
    vsumr_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()