import torch
import triton
import triton.language as tl

@triton.jit
def s341_kernel(
    b_ptr, a_ptr, mask_ptr, prefix_sum_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for conditional array packing.
    Uses prefix sum to determine output positions for positive values.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values from b
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Check condition b > 0
    condition = b_vals > 0.0
    
    # Load prefix sum values to determine output positions
    prefix_vals = tl.load(prefix_sum_ptr + offsets, mask=mask, other=0)
    
    # For elements that satisfy condition, write to output array
    #Prefix sum gives us the 1-based position, subtract 1 for 0-based indexing
    output_mask = mask & condition
    output_offsets = prefix_vals - 1
    
    # Only write if condition is met and we have a valid position
    valid_write_mask = output_mask & (output_offsets >= 0)
    
    tl.store(a_ptr + output_offsets, b_vals, mask=valid_write_mask)

def s341_triton(a, b):
    """
    Triton implementation of TSVC s341 - conditional array packing.
    Optimized using prefix sum to determine output positions in parallel.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = b.numel()
    
    if n_elements == 0:
        return a
    
    # Create mask for condition b > 0
    condition_mask = (b > 0.0).int()
    
    # Compute prefix sum to determine output positions
    # This gives us the cumulative count of positive elements up to each position
    prefix_sum = torch.cumsum(condition_mask, dim=0)
    
    # Clear the output array first
    a.zero_()
    
    # Launch kernel with appropriate block size
    BLOCK_SIZE = 256
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    s341_kernel[grid](
        b, a, condition_mask, prefix_sum,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a