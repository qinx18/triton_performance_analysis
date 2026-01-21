import triton
import triton.language as tl
import torch

@triton.jit
def s311_kernel(a_ptr, result_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Calculate grid position
    pid = tl.program_id(0)
    
    # Calculate offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = current_offsets < N
    
    # Load values from array a
    vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Sum the values in this block
    block_sum = tl.sum(vals)
    
    # Store the partial sum
    tl.store(result_ptr + pid, block_sum)

def s311_triton(a):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid_size, device=a.device, dtype=a.dtype)
    
    # Launch kernel
    grid = (grid_size,)
    s311_kernel[grid](
        a, partial_sums, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results to get final sum
    total_sum = torch.sum(partial_sums)
    
    return total_sum