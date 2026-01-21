import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, partial_sums_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    
    # Apply condition: only sum values > 0, others become 0
    condition = vals > 0.0
    masked_vals = tl.where(condition, vals, 0.0)
    
    block_sum = tl.sum(masked_vals)
    tl.store(partial_sums_ptr + pid, block_sum)

def s3111_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create tensor for partial sums
    partial_sums = torch.zeros(n_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (n_blocks,)
    s3111_kernel[grid](a, partial_sums, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Sum partial results and return as scalar
    return partial_sums.sum()