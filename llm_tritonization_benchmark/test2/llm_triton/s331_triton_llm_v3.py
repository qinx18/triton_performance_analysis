import torch
import triton
import triton.language as tl

@triton.jit
def s331_kernel(a_ptr, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel finds the last index where a[i] < 0
    # We need to process all elements and find the maximum valid index
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements from array a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Check which elements are < 0
    negative_mask = a_vals < 0.0
    
    # Combine with bounds mask
    valid_negative = negative_mask & mask
    
    # For each valid negative element, we want its index
    # We'll use a reduction to find the maximum index
    indices = tl.where(valid_negative, offsets, -1)
    max_idx = tl.max(indices, axis=0)
    
    # Store the result using atomic max to handle multiple blocks
    if max_idx >= 0:
        tl.atomic_max(result_ptr, max_idx)

def s331_triton(a):
    n_elements = a.numel()
    
    # Initialize result tensor with -1 (no negative elements found)
    result = torch.tensor([-1], dtype=torch.int32, device=a.device)
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s331_kernel[grid](
        a, result, n_elements, BLOCK_SIZE
    )
    
    return result.item()