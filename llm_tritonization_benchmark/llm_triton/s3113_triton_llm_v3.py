import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, max_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles one block of elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Compute absolute values
    abs_vals = tl.abs(a_vals)
    
    # Find maximum within this block
    block_max = tl.max(abs_vals, axis=0)
    
    # Store the block maximum
    tl.store(max_ptr + pid, block_max)

@triton.jit
def s3113_final_max_kernel(block_maxes_ptr, final_max_ptr, num_blocks, BLOCK_SIZE: tl.constexpr):
    # Single program to find the final maximum across all blocks
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_blocks
    
    # Load all block maximums
    block_vals = tl.load(block_maxes_ptr + offsets, mask=mask, other=0.0)
    
    # Find global maximum
    global_max = tl.max(block_vals, axis=0)
    
    # Store final result
    tl.store(final_max_ptr, global_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Allocate temporary storage for block maximums
    block_maxes = torch.zeros(num_blocks, device=a.device, dtype=a.dtype)
    
    # Launch first kernel to compute block maximums
    s3113_kernel[(num_blocks,)](
        a, block_maxes, n_elements, BLOCK_SIZE
    )
    
    # Allocate storage for final result
    final_max = torch.zeros(1, device=a.device, dtype=a.dtype)
    
    # Find next power of 2 for final reduction block size
    final_block_size = 1
    while final_block_size < num_blocks:
        final_block_size *= 2
    final_block_size = min(final_block_size, 1024)
    
    # Launch second kernel to find global maximum
    s3113_final_max_kernel[(1,)](
        block_maxes, final_max, num_blocks, final_block_size
    )
    
    return final_max.item()