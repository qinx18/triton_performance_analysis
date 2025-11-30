import torch
import triton
import triton.language as tl

@triton.jit
def s314_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Each program handles one block of elements
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load block of data
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find maximum in this block
    block_max = tl.max(a_vals, axis=0)
    
    # Store the block maximum
    tl.store(result_ptr + pid, block_max)

def s314_triton(a):
    n_elements = a.shape[0]
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # First pass: compute block-wise maxima
    block_maxes = torch.empty(num_blocks, device=a.device, dtype=a.dtype)
    
    s314_kernel[(num_blocks,)](
        a, block_maxes, n_elements, BLOCK_SIZE
    )
    
    # Second pass: find global maximum from block maxima
    # Start with a[0] as required by the original algorithm
    x = a[0].clone()
    
    # Compare with all block maxima to find global maximum
    global_max = torch.max(block_maxes)
    
    # Return the maximum of a[0] and the computed global maximum
    x = torch.max(x, global_max)
    
    return x