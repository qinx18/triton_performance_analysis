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
    """
    Triton kernel for s311 - sum reduction with block-level optimization
    Uses block-wise reduction to minimize memory accesses
    """
    pid = tl.program_id(axis=0)
    
    # Calculate offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load data with masking for edge cases
    mask = offsets < n_elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Perform block-wise sum reduction
    block_sum = tl.sum(a_vals)
    
    # Atomic add to accumulate partial sums from all blocks
    tl.atomic_add(sum_ptr, block_sum)

def s311_triton(a):
    """
    Triton implementation of TSVC s311 - sum reductions
    
    Optimizations:
    - Block-wise parallel reduction across GPU blocks
    - Coalesced memory access with proper masking
    - Atomic accumulation for final sum
    """
    a = a.contiguous()
    n_elements = a.numel()
    
    # Allocate output for sum (initialized to zero)
    sum_result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    # Choose block size for optimal occupancy
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s311_kernel[grid](
        a_ptr=a,
        sum_ptr=sum_result,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Return the input array unchanged (matching baseline behavior)
    return a