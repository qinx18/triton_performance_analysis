import triton
import triton.language as tl
import torch

@triton.jit
def s3113_kernel(a_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Load block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values and compute absolute values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    abs_vals = tl.abs(a_vals)
    
    # Find maximum in this block
    block_max = tl.max(abs_vals)
    
    # Store block maximum
    tl.store(output_ptr + tl.program_id(0), block_max)

def s3113_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # First reduction: find max per block
    block_maxes = torch.zeros(num_blocks, device=a.device, dtype=a.dtype)
    
    grid = (num_blocks,)
    s3113_kernel[grid](a, block_maxes, n_elements, BLOCK_SIZE)
    
    # Second reduction: find global maximum
    max_val = torch.max(block_maxes).item()
    
    return max_val