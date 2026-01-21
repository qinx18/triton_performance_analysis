import triton
import triton.language as tl
import torch

@triton.jit
def s3111_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Load values
    vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Apply condition: a[i] > 0.0 and sum only those values
    conditional_vals = tl.where(vals > 0.0, vals, 0.0)
    block_sum = tl.sum(conditional_vals)
    
    # Store partial sum
    tl.store(result_ptr + pid, block_sum)

def s3111_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 1024
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (grid_size,)
    s3111_kernel[grid](a, partial_sums, N, BLOCK_SIZE=BLOCK_SIZE)
    
    # Final reduction
    return partial_sums.sum().item()