import torch
import triton
import triton.language as tl

@triton.jit
def s3111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_sum = 0.0
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        condition = vals > 0.0
        masked_vals = tl.where(condition, vals, 0.0)
        block_sum += tl.sum(masked_vals, axis=0)
    
    tl.store(result_ptr, block_sum)

def s3111_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 1024
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    s3111_kernel[(1,)](
        a, result, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()