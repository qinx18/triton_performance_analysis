import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, partial_sums_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    condition_mask = vals > 0.0
    filtered_vals = tl.where(condition_mask & mask, vals, 0.0)
    block_sum = tl.sum(filtered_vals)
    
    tl.store(partial_sums_ptr + pid, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    s3111_kernel[(num_blocks,)](
        a, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return partial_sums.sum().item()