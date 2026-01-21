import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, partial_sum_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    condition = vals > 0.0
    conditional_vals = tl.where(condition, vals, 0.0)
    block_sum = tl.sum(conditional_vals)
    
    tl.store(partial_sum_ptr + pid, block_sum)

def s3111_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    grid = (num_blocks,)
    s3111_kernel[grid](a, partial_sums, n, BLOCK_SIZE)
    
    total_sum = partial_sums.sum().item()
    return total_sum