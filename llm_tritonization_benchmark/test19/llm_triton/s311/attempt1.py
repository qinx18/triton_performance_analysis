import triton
import triton.language as tl
import torch

@triton.jit
def s311_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    total_sum = 0.0
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        block_sum = tl.sum(vals)
        total_sum += block_sum
    
    if tl.program_id(0) == 0:
        tl.store(result_ptr, total_sum)

def s311_triton(a):
    n = a.shape[0]
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s311_kernel[grid](
        a, result, n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()