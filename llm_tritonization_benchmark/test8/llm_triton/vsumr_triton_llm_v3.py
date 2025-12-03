import torch
import triton
import triton.language as tl

@triton.jit
def vsumr_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    partial_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        partial_sum += vals
    
    total_sum = tl.sum(partial_sum, axis=0)
    
    if tl.program_id(0) == 0:
        tl.store(result_ptr, total_sum)

def vsumr_triton(a):
    BLOCK_SIZE = 1024
    n_elements = a.shape[0]
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    
    vsumr_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()