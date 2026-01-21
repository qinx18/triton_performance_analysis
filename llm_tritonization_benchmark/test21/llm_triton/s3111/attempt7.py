import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    condition = vals > 0.0
    
    # Apply condition and sum
    conditional_vals = tl.where(condition, vals, 0.0)
    block_sum = tl.sum(conditional_vals)
    
    # Store partial sum
    tl.store(result_ptr + pid, block_sum)

def s3111_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 1024
    
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    grid = (num_blocks,)
    s3111_kernel[grid](a, partial_sums, N, BLOCK_SIZE=BLOCK_SIZE)
    
    return partial_sums.sum().item()