import torch
import triton
import triton.language as tl

@triton.jit
def s3113_kernel(
    a_ptr,
    max_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for finding maximum absolute value using parallel reduction.
    Each block processes BLOCK_SIZE elements and finds local max, then reduces globally.
    """
    pid = tl.program_id(axis=0)
    
    # Calculate block offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load data with masking (use 0.0 for out-of-bounds elements)
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Compute absolute values
    abs_vals = tl.abs(a_vals)
    
    # Find maximum within this block using reduction
    block_max = tl.max(abs_vals, axis=0)
    
    # Atomic maximum to find global maximum across all blocks
    tl.atomic_max(max_ptr, block_max)

def s3113_triton(a):
    """
    Triton implementation of TSVC s3113 - finding maximum absolute value.
    Uses parallel reduction across GPU blocks for efficient computation.
    """
    a = a.contiguous()
    
    # Initialize output tensor for maximum value
    max_val = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    n_elements = a.numel()
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s3113_kernel[grid](
        a,
        max_val,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a