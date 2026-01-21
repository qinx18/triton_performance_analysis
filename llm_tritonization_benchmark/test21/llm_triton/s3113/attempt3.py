import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, partial_max_ptr, n, BLOCK_SIZE: tl.constexpr):
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n
    
    # Load values and compute absolute values
    vals = tl.load(a_ptr + indices, mask=mask, other=0.0)
    abs_vals = tl.abs(vals)
    
    # Set masked-out values to 0 (they won't affect max since we start with abs(a[0]))
    masked_abs_vals = tl.where(mask, abs_vals, 0.0)
    
    # Find maximum within this block
    block_max = tl.max(masked_abs_vals)
    
    # Store partial result
    tl.store(partial_max_ptr + program_id, block_max)

@triton.jit
def s3113_reduce_kernel(partial_max_ptr, output_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    # Load partial results
    partial_vals = tl.load(partial_max_ptr + offsets, mask=mask, other=0.0)
    
    # Find global maximum
    global_max = tl.max(partial_vals)
    
    # Store final result (only first thread)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, global_max)

def s3113_triton(a):
    n = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Get abs(a[0]) as starting value
    max_val = torch.abs(a[0:1]).clone()
    
    if n == 1:
        return max_val.item()
    
    # Allocate space for partial results
    partial_max = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel to compute partial maximums
    grid = (num_blocks,)
    s3113_kernel[grid](a, partial_max, n, BLOCK_SIZE=BLOCK_SIZE)
    
    # Reduce partial results
    if num_blocks == 1:
        final_max = torch.maximum(max_val, partial_max)
    else:
        output = torch.zeros(1, dtype=a.dtype, device=a.device)
        reduce_blocks = triton.cdiv(num_blocks, BLOCK_SIZE)
        grid = (reduce_blocks,)
        s3113_reduce_kernel[grid](partial_max, output, num_blocks, BLOCK_SIZE=BLOCK_SIZE)
        final_max = torch.maximum(max_val, output)
    
    return final_max.item()