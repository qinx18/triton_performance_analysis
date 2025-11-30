import torch
import triton
import triton.language as tl

@triton.jit
def s311_kernel(
    a_ptr,
    sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load values from array a
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Sum the values in this block
    block_sum = tl.sum(a_vals)
    
    # Store the block sum (this will be accumulated later)
    if pid == 0:
        tl.store(sum_ptr, block_sum)
    else:
        # Atomic add for accumulation across blocks
        tl.atomic_add(sum_ptr, block_sum)

def s311_triton(a):
    n_elements = a.numel()
    
    # Initialize sum to zero
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Calculate grid size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s311_kernel[grid](
        a,
        sum_result,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return sum_result.item()