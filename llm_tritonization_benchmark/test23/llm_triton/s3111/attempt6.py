import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, result_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    total_sum = 0.0
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        positive_mask = vals > 0.0
        filtered_vals = tl.where(positive_mask, vals, 0.0)
        block_sum = tl.sum(filtered_vals, axis=0)
        
        total_sum += block_sum
    
    tl.store(result_ptr, total_sum)

def s3111_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s3111_kernel[grid](a, result, n, BLOCK_SIZE=BLOCK_SIZE)
    
    return result.item()