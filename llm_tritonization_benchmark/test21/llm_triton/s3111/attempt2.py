import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_sum = tl.zeros((1,), dtype=tl.float32)
    for block_start in range(0, N, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < N
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Apply condition: if a[i] > 0
        condition_mask = vals > 0.0
        conditional_vals = tl.where(condition_mask, vals, 0.0)
        
        block_sum += tl.sum(conditional_vals)
    
    tl.store(result_ptr, block_sum)

def s3111_triton(a):
    N = a.shape[0]
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s3111_kernel[grid](
        a, result, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return result.item()